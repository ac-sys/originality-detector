"""Feature extractors for v2 detection experiments (CPU). Each returns a dict of floats per text."""
import torch, numpy as np, math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig, AutoModel
import os; torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
MAXTOK = 320

class LMPair:
    """Observer (base) + performer (instruct) causal LMs sharing a tokenizer.
    Gives: mean log-prob, mean entropy, Binoculars score (Hans et al. 2024) and analytic
    Fast-DetectGPT curvature (Bao et al. 2024) with reference=performer, scoring=observer, and
    the single-model variant (reference=scoring=observer)."""
    def __init__(self, base, inst, prefix):
        self.tok = AutoTokenizer.from_pretrained(base); self.p = prefix
        self.obs = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).eval()
        self.perf = AutoModelForCausalLM.from_pretrained(inst, torch_dtype=torch.float32).eval() if inst else None
    @torch.no_grad()
    def token_stats(self, text):
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=MAXTOK, add_special_tokens=False).input_ids
        if self.tok.bos_token_id is not None: ids = torch.cat([torch.tensor([[self.tok.bos_token_id]]), ids], 1)
        lo = torch.log_softmax(self.obs(ids).logits[0, :-1].float(), -1); y = ids[0, 1:]
        lp_o = lo.gather(1, y[:, None])[:, 0]
        po = lo.exp(); ent_o = -(po * lo).sum(-1)
        out = {"lp": lp_o, "ent": ent_o}
        # single-model fast-detectgpt per token: (lp - mu)/sd where mu=E_p[lp], var=E_p[lp^2]-mu^2
        mu = (po * lo).sum(-1); var = (po * lo * lo).sum(-1) - mu ** 2
        out["fd_mu"], out["fd_var"] = mu, var
        if self.perf is not None:
            lpf = torch.log_softmax(self.perf(ids).logits[0, :-1].float(), -1); pf = lpf.exp()
            out["xent"] = -(pf * lo).sum(-1)                   # binoculars cross-perplexity per token
            mu2 = (pf * lo).sum(-1); var2 = (pf * lo * lo).sum(-1) - mu2 ** 2
            out["fd2_mu"], out["fd2_var"] = mu2, var2
        return out
    def features(self, text):
        s = self.token_stats(text); p = self.p; f = {}
        f[p + "lp"] = s["lp"].mean().item(); f[p + "ent"] = s["ent"].mean().item()
        f[p + "lp_std"] = s["lp"].std().item()
        f[p + "fastdetect1"] = ((s["lp"].sum() - s["fd_mu"].sum()) / s["fd_var"].sum().clamp_min(1e-6).sqrt()).item()
        if "xent" in s:
            f[p + "binoculars"] = (-s["lp"].mean() / s["xent"].mean()).item()   # low => AI
            f[p + "fastdetect2"] = ((s["lp"].sum() - s["fd2_mu"].sum()) / s["fd2_var"].sum().clamp_min(1e-6).sqrt()).item()
        return f
    def span_scores(self, text, spans):
        """Per-span (sentence group) Fast-DetectGPT/binoculars using char offsets."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)
        return enc

class Classifier:
    def __init__(self, hf_id, name, ai_label=None, maxlen=512):
        self.tok = AutoTokenizer.from_pretrained(hf_id); self.n = name; self.maxlen = maxlen
        self.m = AutoModelForSequenceClassification.from_pretrained(hf_id).eval()
        lab = {v.lower(): int(k) for k, v in self.m.config.id2label.items()}
        self.ai = lab.get((ai_label or "").lower(), 1); self.nl = self.m.config.num_labels
    @torch.no_grad()
    def batch(self, texts):
        x = self.tok(texts, truncation=True, max_length=self.maxlen, padding=True, return_tensors="pt")
        lg = self.m(**x).logits.float()
        if self.nl == 1: return lg[:, 0].tolist()
        return (lg[:, self.ai] - torch.logsumexp(torch.cat([lg[:, :self.ai], lg[:, self.ai+1:]], 1), 1)).tolist()

class Desklib:
    """desklib/ai-text-detector-v1.01 (MIT): DeBERTa-v3-large + mean pooling + linear head (custom class)."""
    def __init__(self, hf_id="desklib/ai-text-detector-v1.01", maxlen=512):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import os
        self.tok = AutoTokenizer.from_pretrained(hf_id); cfg = AutoConfig.from_pretrained(hf_id)
        self.enc = AutoModel.from_config(cfg); self.head = torch.nn.Linear(cfg.hidden_size, 1); self.maxlen = maxlen
        path = hf_hub_download(hf_id, "model.safetensors"); sd = load_file(path)
        self.enc.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
        self.head.load_state_dict({k[len("classifier."):]: v for k, v in sd.items() if k.startswith("classifier.")})
        self.enc.eval()
    @torch.no_grad()
    def batch(self, texts):
        x = self.tok(texts, truncation=True, max_length=self.maxlen, padding=True, return_tensors="pt")
        h = self.enc(**x).last_hidden_state; m = x["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp_min(1e-9))[:, 0].tolist()
