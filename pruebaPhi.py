# ==== Inferencia y evaluación (4 prompts) con OpenVINO (Intel GPU Iris / CPU) ====
# - Entrada:
#     - Notas: C:\Users\hered\Desktop\TFM\TFM\IMC2\eval_imc_fullnotes.csv  (cols: patient_id + texto)
#     - GT   : C:\Users\hered\Desktop\TFM\TFM\IMC2\valid_imc.csv          (cols: patient_id, height_m, weight_kg, BMI, ...)
# - Salida:
#     - ./outputs/pred_*.csv  (por prompt)
#     - ./outputs/eval_*.csv  (join con GT + métricas)

import os, sys, json, warnings, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

#  Dependencias  
REQ = [
    "numpy<2.1",
    "pandas",
    "tqdm",
    "transformers==4.46.2",
    "accelerate>=0.34",
    "huggingface_hub>=0.24",
    "sacremoses",
    "sentencepiece",
    "optimum-intel[openvino]==1.26.0",
    "openvino>=2025.1.0",
]
def ensure(pkgs):
    need=[]
    for spec in pkgs:
        name = spec.split("==")[0].split(">=")[0].split("<")[0].split("[")[0]
        try: __import__(name.replace("-","_"))
        except Exception: need.append(spec)
    if need:
        print(" Instalando:", need)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + need)
ensure(REQ)

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

#  Rutas / Config 
NOTES_CSV = r"C:\Users\hered\Desktop\TFM\TFM\IMC2\eval_imc_fullnotes.csv"
GT_CSV    = r"C:\Users\hered\Desktop\TFM\TFM\IMC2\valid_imc.csv"
OUT_DIR   = str(Path("./outputs").resolve())

# CAMBIO SOLO DEL MODELO - HIPERPARÁMETROS IDÉNTICOS
MODEL_ID  = os.getenv("MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

# Inferencia - HIPERPARÁMETROS EXACTAMENTE IGUALES
ATTEMPTS_PER_WIN = 4
N_WINDOWS_MAX    = 8
TEMP, TOP_P      = 0.4, 0.95
MAX_NEW          = 180

# Plausibilidad / tolerancias - estos hiperparámetros permanecen iguales en todos los experimentos
H_MIN, H_MAX = 1.2, 2.2
W_MIN, W_MAX = 30, 300
BMI_MIN, BMI_MAX = 10, 80
BMI_TOL = 0.5
PROTOTYPES = {1.60, 1.70, 1.73, 1.75, 65.0, 70.0, 72.5, 75.0, 22.49, 24.2, 25.0}

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

#  Carga de datos 
def load_notes(csv_path):
    # lector robusto (comillas/líneas largas)
    df = pd.read_csv(csv_path, dtype={"patient_id": str})
    # detectar columna de texto
    cand = [c for c in df.columns if str(c).lower() in {"patient","note","text","note_text","full_note"}]
    if not cand:
        # si sólo hay 2 columnas, toma la segunda como texto
        if len(df.columns) >= 2:
            cand = [df.columns[1]]
        else:
            raise ValueError(f"No encuentro columna de texto en {csv_path}. Cols: {df.columns.tolist()}")
    txt_col = cand[0]
    # normalizar nombres
    df = df.rename(columns={txt_col:"patient"})
    assert "patient_id" in df.columns, "Falta 'patient_id' en notas"
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["patient"]    = df["patient"].astype(str)
    return df[["patient_id","patient"]]

def load_gt(csv_path):
    df = pd.read_csv(csv_path, dtype={"patient_id": str})
    assert "patient_id" in df.columns, "Falta 'patient_id' en GT"
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    for c in ["height_m","weight_kg","BMI"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

notes = load_notes(NOTES_CSV)
gt    = load_gt(GT_CSV)
# NOTA: He comentado la limitación a 20 registros para procesar todos los datos
notes = notes.head(20)
print(f"Notas: {len(notes)} | GT: {len(gt)} | Intersección: {len(set(notes.patient_id)&set(gt.patient_id))}")

#  Modelo OpenVINO (Intel GPU → CPU fallback) 
def get_ov_model_and_tokenizer(model_id: str, device_pref: str = "GPU"):
    print(f"\nCargando tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    device = "GPU" if device_pref.upper()=="GPU" else "CPU"
    print(f"Cargando modelo OpenVINO en {device} (exporta si hace falta)…")
    
    try:
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            device=device,
            compile=True,
            trust_remote_code=True,
            ov_config={"CACHE_DIR": str(Path(OUT_DIR)/"ov_cache")},
        )
        
        if getattr(ov_model.config, "pad_token_id", None) is None and tok.eos_token_id is not None:
            ov_model.config.pad_token_id = tok.eos_token_id

        # Test → si falla en GPU, reintenta CPU
        try:
            test_ids = tok("ok", return_tensors="pt").input_ids
            _ = ov_model.generate(test_ids, max_new_tokens=1)
            print(f" Modelo operativo en {device}")
        except Exception as e:
            if device == "GPU":
                print(f" Falló en GPU ({e}). Reintentando en CPU…")
                ov_model = OVModelForCausalLM.from_pretrained(
                    model_id, export=True, device="CPU", compile=True, trust_remote_code=True,
                    ov_config={"CACHE_DIR": str(Path(OUT_DIR)/"ov_cache")},
                )
                print(" Modelo operativo en CPU")
            else:
                raise
                
    except Exception as e:
        print(f" Error cargando el modelo: {e}")
        print(" Intentando con modelo de respaldo...")
        # Fallback a un modelo aún más pequeño
        model_id = "microsoft/DialoGPT-medium"
        print(f" Cambiando a modelo: {model_id}")
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok.pad_token = tok.eos_token
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id, export=True, device="CPU", compile=True, trust_remote_code=True
        )

    def llm_generate(prompt: str, max_new=MAX_NEW, temperature=TEMP, top_p=TOP_P, do_sample=True):
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        out_ids = ov_model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
            eos_token_id=(tok.eos_token_id or ov_model.config.eos_token_id),
            pad_token_id=(ov_model.config.pad_token_id or tok.eos_token_id),
        )
        return tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def apply_chat_template(system_text: str, user_text: str):
        """Usa chat_template si existe; si no, fallback Q/A estable."""
        has_apply = hasattr(tok, "apply_chat_template")
        has_template = bool(getattr(tok, "chat_template", None))
        if has_apply and has_template:
            try:
                return tok.apply_chat_template(
                    [{"role":"system","content":system_text},
                     {"role":"user","content":user_text}],
                    tokenize=False, add_generation_prompt=True,
                )
            except:
                # Fallback si el template falla
                pass
        # Fallback para modelos no-instruct
        return (
            "System:\n" + system_text.strip() + "\n\n"
            "User:\n"   + user_text.strip()   + "\n\n"
            "Assistant:\n"
        )

    return tok, llm_generate, apply_chat_template

print(f"Cargando modelo OpenVINO: {MODEL_ID}")
tokenizer, llm_generate, apply_chat_template = get_ov_model_and_tokenizer(MODEL_ID, device_pref="GPU")

#  Ventanas / utils numéricas 
WIN, STRIDE = 1100, 800
UNIT_TOKENS = [" cm"," m","meter","metre","ft"," in","inch","kg"," lb","lbs","pound"," stone"," st","bmi","BMI","weight","height"]

def window_iter(text, win=WIN, stride=STRIDE):
    t = str(text); n = len(t)
    if n <= win:
        yield 0, t; return
    for i in range(0, n, stride):
        yield i, t[i:i+win]
        if i+win >= n: break

def has_unit_token(s: str):
    sl = (s or "").lower()
    return any(tok.strip().lower() in sl for tok in UNIT_TOKENS)

def is_num(x):
    try: return np.isfinite(float(x))
    except: return False

def clip_plausible(h, w, b):
    try:
        if is_num(h) and not (H_MIN <= float(h) <= H_MAX): h = None
    except: h = None
    try:
        if is_num(w) and not (W_MIN <= float(w) <= W_MAX): w = None
    except: w = None
    try:
        if is_num(b) and not (BMI_MIN <= float(b) <= BMI_MAX): b = None
    except: b = None
    return h, w, b

def recompute_bmi(h, w):
    try:
        h = float(h); w = float(w)
        if h > 0: return round(w/(h*h), 2)
    except: pass
    return None

def proto_penalty(x):
    try:
        if x is None: return 0.0
        v = round(float(x), 2)
        return -0.35 if v in PROTOTYPES else 0.0
    except: return 0.0

#  4 PROMPTS 
SYSTEM_SIMPLE = (
    "You are a careful clinical extractor. From the GIVEN WINDOW ONLY, return STRICT JSON with normalized SI values:\n"
    "{ \"height_m\": <float|null>, \"weight_kg\": <float|null>, \"bmi\": <float|null> }\n"
    "Rules: Use ONLY numbers present; convert units to SI; if either height or weight is missing, bmi=null. Output JSON only."
)

SYSTEM_STRICT = (
    "You are a clinical extractor and verifier. From the GIVEN WINDOW ONLY, return STRICT JSON:\n"
    "{ \"height_m\": <float|null>, \"weight_kg\": <float|null>, \"bmi\": <float|null> }\n"
    "• Use ONLY numbers with explicit units (cm/m/ft-in → m; kg/lb/stone → kg).\n"
    "• Plausibility: 1.20–2.20 m, 30–300 kg, 10–80 BMI.\n"
    "• If both H & W exist, compute bmi=kg/(m^2) (2 decimals) and prefer it over conflicting BMI text.\n"
    "• If inconsistent, set bmi=null. JSON only."
)

SYSTEM_FEWSHOT = SYSTEM_SIMPLE
FEW_SHOTS = [
    ("A 60-year-old woman, height 165 cm and weight 68 kg.",
     "{\"height_m\": 1.65, \"weight_kg\": 68.0, \"bmi\": 24.98}"),
    ("Male, 1.80 m, 90 kg; BMI not explicitly stated in text.",
     "{\"height_m\": 1.80, \"weight_kg\": 90.0, \"bmi\": 27.78}"),
    ("Patient reports good energy. No numeric measurements present.",
     "{\"height_m\": null, \"weight_kg\": null, \"bmi\": null}"),
]

#  Prompt builder / parser / scoring 
def build_prompt(system_text: str, window_text: str):
    msgs = [{"role":"system","content":system_text},
            {"role":"user","content":"NOTE WINDOW:\n"+window_text+"\n\nJSON ONLY"}]
    if hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer,"chat_template",None)):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return (
        "System:\n"+system_text+"\n\n"+
        "User:\nNOTE WINDOW:\n"+window_text+"\n\nJSON ONLY\n\n"+
        "Assistant:\n"
    )

def safe_json(text: str, expect="triplet"):
    if not text: return None
    s = text.strip()
    if s.startswith("```"):
        try:
            s = s.split("```", 1)[-1]
            if "```" in s: s = s.split("```",1)[0]
        except: pass
    a, b = s.find("{"), s.rfind("}")
    if a!=-1 and b!=-1 and b>a: s = s[a:b+1]
    s = s.replace("None","null").replace("NaN","null").replace(",}", "}")
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict): return None
        if expect=="bmi_only":
            return obj if "bmi" in obj else None
        else:
            return obj if {"height_m","weight_kg","bmi"}.issubset(obj.keys()) else None
    except: return None

def score_triplet(h, w, b):
    s = 0.0
    if is_num(h): s += 1.0
    if is_num(w): s += 1.0
    if is_num(h) and is_num(w):
        b2 = recompute_bmi(h, w)
        if is_num(b2): s += 0.7
        if is_num(b) and abs(float(b) - float(b2)) <= BMI_TOL: s += 0.4
    elif is_num(b):
        s += 0.2
    s += proto_penalty(h) + proto_penalty(w) + proto_penalty(b)
    return s

#  Ejecutores (simple/estricto/fewshot) 
def run_triplet(note_text: str, system_text: str):
    wins = [(s, c) for s, c in window_iter(note_text)]
    wins = sorted(wins, key=lambda x: int(not has_unit_token(x[1])))
    best={"h":None,"w":None,"b":None,"score":-1e9}
    for _, chunk in wins[:N_WINDOWS_MAX]:
        prompt = build_prompt(system_text, chunk)
        for _ in range(ATTEMPTS_PER_WIN):
            raw = llm_generate(prompt)
            obj = safe_json(raw, expect="triplet")
            if obj is None: continue
            h,w,b = obj.get("height_m"), obj.get("weight_kg"), obj.get("bmi")
            try: h=float(h)
            except: h=None
            try: w=float(w)
            except: w=None
            try: b=float(b)
            except: b=None
            h,w,b = clip_plausible(h,w,b)
            sc = score_triplet(h,w,b)
            if sc>best["score"]:
                best={"h":h,"w":w,"b":b,"score":sc}
        if is_num(best["h"]) and is_num(best["w"]):
            break
    H = round(float(best["h"]),2) if is_num(best["h"]) else None
    W = round(float(best["w"]),1) if is_num(best["w"]) else None
    B_from = recompute_bmi(H,W) if (is_num(H) and is_num(W)) else None
    B = B_from if is_num(B_from) else (round(float(best["b"]),2) if is_num(best["b"]) else None)
    return H,W,B_from,B

def run_fewshot(note_text: str, system_text: str):
    wins = [(s, c) for s, c in window_iter(note_text)]
    wins = sorted(wins, key=lambda x: int(not has_unit_token(x[1])))
    best={"h":None,"w":None,"b":None,"score":-1e9}
    for _, chunk in wins[:N_WINDOWS_MAX]:
        # ensamblado few-shot si no hay chat_template
        if hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer,"chat_template",None)):
            prompt = build_prompt(system_text, chunk)
        else:
            examples = []
            for ex_in, ex_out in FEW_SHOTS:
                examples.append(f"Input:\n{ex_in}\nOutput:\n{ex_out}\n")
            ex_block = "\n".join(examples)
            prompt = (
                "Task:\n"+system_text.strip()+"\n\n"+
                "Examples:\n"+ex_block+"\n"+
                "Input:\nNOTE WINDOW:\n"+chunk+"\n\nJSON ONLY\n"+
                "Output:\n"
            )
        for _ in range(ATTEMPTS_PER_WIN):
            raw = llm_generate(prompt)
            obj = safe_json(raw, expect="triplet")
            if obj is None: continue
            h,w,b = obj.get("height_m"), obj.get("weight_kg"), obj.get("bmi")
            try: h=float(h)
            except: h=None
            try: w=float(w)
            except: w=None
            try: b=float(b)
            except: b=None
            h,w,b = clip_plausible(h,w,b)
            sc = score_triplet(h,w,b)
            if sc>best["score"]:
                best={"h":h,"w":w,"b":b,"score":sc}
        if is_num(best["h"]) and is_num(best["w"]):
            break
    H = round(float(best["h"]),2) if is_num(best["h"]) else None
    W = round(float(best["w"]),1) if is_num(best["w"]) else None
    B_from = recompute_bmi(H,W) if (is_num(H) and is_num(W)) else None
    B = B_from if is_num(B_from) else (round(float(best["b"]),2) if is_num(best["b"]) else None)
    return H,W,B_from,B

#  Encadenado (SPAN -> NORM) 
def chat_prompt(system, user):
    return apply_chat_template(system, user)

def run_chain_on_window(window_text: str):
    SYS_SPAN = (
        "You are a clinical span finder. From the NOTE WINDOW, pick ONLY the earliest sentence that "
        "contains tokens/units for height or weight or BMI, and return STRICT JSON:\n"
        "{ \"sentence\": <string>, \"height_span\": <string|null>, \"weight_span\": <string|null>, \"bmi_span\": <string|null> }\n"
        "Spans must be exact substrings and include units when applicable. JSON only."
    )
    SYS_NORM = (
        "You are a clinical normalizer and calculator. Given the chosen sentence and spans, return STRICT JSON:\n"
        "{ \"height_m\": <float|null>, \"weight_kg\": <float|null>, \"bmi\": <float|null>, "
        "\"bmi_source\": <\"from_text\"|\"from_hw\"|null>, \"check\": <\"ok\"|\"mismatch\"|\"insufficient\"> }\n"
        "Normalize units; if both H & W exist, COMPUTE bmi=kg/(m^2) (2 decimals). Prefer computed BMI if conflicting."
    )
    span_raw = llm_generate(chat_prompt(SYS_SPAN, f"NOTE WINDOW:\n{window_text}\n\nJSON ONLY"))
    # permitimos que el primer paso no sea tripleta; sólo extraemos sentence/spans si vienen
    span_obj = None
    try:
        s = span_raw.strip()
        a,b = s.find("{"), s.rfind("}")
        if a!=-1 and b!=-1 and b>a:
            span_obj = json.loads(s[a:b+1])
    except Exception:
        span_obj = {}
    if not isinstance(span_obj, dict): span_obj = {}

    norm_user = json.dumps({
        "sentence": span_obj.get("sentence",""),
        "height_span": span_obj.get("height_span"),
        "weight_span": span_obj.get("weight_span"),
        "bmi_span": span_obj.get("bmi_span")
    }, ensure_ascii=False)
    norm_raw = llm_generate(chat_prompt(SYS_NORM, norm_user + "\n\nJSON ONLY"))
    norm_obj = safe_json(norm_raw, expect="triplet") or {}
    return {
        "sentence": span_obj.get("sentence",""),
        "height_m": norm_obj.get("height_m"),
        "weight_kg": norm_obj.get("weight_kg"),
        "bmi": norm_obj.get("bmi"),
        "bmi_source": norm_obj.get("bmi_source"),
        "check": norm_obj.get("check")
    }

def run_chain_on_note(note_text: str, attempts_per_win=2, n_windows_max=6):
    cands=[]
    for _, chunk in list(window_iter(note_text))[:n_windows_max]:
        for _ in range(attempts_per_win):
            c = run_chain_on_window(chunk)
            if isinstance(c, dict): cands.append(c)
    def to_float(x):
        try: return float(x)
        except: return None
    # juez sencillo: prioriza check ok, si no más campos no nulos
    best = None
    def nonnulls(c): return sum([c.get("height_m") is not None, c.get("weight_kg") is not None, c.get("bmi") is not None])
    for c in cands:
        if c.get("check") == "ok":
            best = c; break
    if best is None and cands:
        best = sorted(cands, key=lambda x: nonnulls(x), reverse=True)[0]
    if not best: best={}
    return {
        "height_m_pred": to_float(best.get("height_m")),
        "weight_kg_pred": to_float(best.get("weight_kg")),
        "BMI_pred_raw":   to_float(best.get("bmi")),
        "bmi_source":     best.get("bmi_source"),
        "check":          best.get("check")
    }

#  Orquestación (4 prompts) 
PROMPTS = {
    "v1_simple":    SYSTEM_SIMPLE,
    "v2_estricto":  SYSTEM_STRICT,
    "v3_fewshot":   SYSTEM_FEWSHOT,
    "v4_encadenado": None,   # chain
}

def save_rows(rows, out_csv):
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f" Guardado: {out_csv}")

ALL_PRED_PATHS = []

for pid_name, system_text in PROMPTS.items():
    out_csv = f"{OUT_DIR}/pred_{pid_name}_{MODEL_ID.split('/')[-1]}_n{len(notes)}.csv"
    rows=[]
    if pid_name == "v4_encadenado":
        for _, r in tqdm(notes.iterrows(), total=len(notes), desc=f"Inferencia {pid_name}"):
            pid, note = r["patient_id"], str(r["patient"])
            out = run_chain_on_note(note, attempts_per_win=2, n_windows_max=6)
            rows.append({
                "patient_id": pid, "note_len": len(note), "prompt_id": pid_name, "model_used": MODEL_ID,
                "height_m_pred": (out["height_m_pred"] if out["height_m_pred"] is not None else np.nan),
                "weight_kg_pred": (out["weight_kg_pred"] if out["weight_kg_pred"] is not None else np.nan),
                "BMI_pred_raw":   (out["BMI_pred_raw"] if out["BMI_pred_raw"] is not None else np.nan),
                "BMI_from_pred_hw": np.nan,
                "bmi_source": out.get("bmi_source"), "check": out.get("check")
            })
    elif pid_name == "v3_fewshot":
        for _, r in tqdm(notes.iterrows(), total=len(notes), desc=f"Inferencia {pid_name}"):
            pid, note = r["patient_id"], str(r["patient"])
            H,W,B_from,B = run_fewshot(note, system_text)
            rows.append({
                "patient_id": pid, "note_len": len(note), "prompt_id": pid_name, "model_used": MODEL_ID,
                "height_m_pred": H if H is not None else np.nan,
                "weight_kg_pred": W if W is not None else np.nan,
                "BMI_from_pred_hw": B_from if B_from is not None else np.nan,
                "BMI_pred_raw": B if B is not None else np.nan
            })
    else:
        for _, r in tqdm(notes.iterrows(), total=len(notes), desc=f"Inferencia {pid_name}"):
            pid, note = r["patient_id"], str(r["patient"])
            H,W,B_from,B = run_triplet(note, system_text)
            rows.append({
                "patient_id": pid, "note_len": len(note), "prompt_id": pid_name, "model_used": MODEL_ID,
                "height_m_pred": H if H is not None else np.nan,
                "weight_kg_pred": W if W is not None else np.nan,
                "BMI_from_pred_hw": B_from if B_from is not None else np.nan,
                "BMI_pred_raw": B if B is not None else np.nan
            })
    save_rows(rows, out_csv)
    ALL_PRED_PATHS.append(out_csv)

#  Evaluación contra GT 
def eval_and_save(pred_path, gt_df):
    pred = pd.read_csv(pred_path, dtype={"patient_id": str})
    pred["patient_id"] = pred["patient_id"].astype(str).str.strip()
    join = pred.merge(gt_df[["patient_id","height_m","weight_kg","BMI"]], on="patient_id", how="inner")
    # métricas
    def mae(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(np.abs(a[m]-b[m]))) if m.any() else np.nan
    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else np.nan

    # BMI "mejor" del modelo: prioriza BMI_from_pred_hw si está; si no, BMI_pred_raw
    join["BMI_model_best"] = np.where(
        pd.notnull(join.get("BMI_from_pred_hw", np.nan)),
        join["BMI_from_pred_hw"],
        join.get("BMI_pred_raw", np.nan)
    )

    # arrays numéricos
    bmi_gt   = pd.to_numeric(join["BMI"], errors="coerce").to_numpy(dtype=float)
    bmi_raw  = pd.to_numeric(join.get("BMI_pred_raw", np.nan), errors="coerce").to_numpy(dtype=float)
    bmi_hw   = pd.to_numeric(join.get("BMI_from_pred_hw", np.nan), errors="coerce").to_numpy(dtype=float)
    bmi_best = pd.to_numeric(join.get("BMI_model_best", np.nan), errors="coerce").to_numpy(dtype=float)

    metrics = {
        "n_overlap": int(len(join)),
        "MAE_BMI_raw":  mae(bmi_raw, bmi_gt),
        "RMSE_BMI_raw": rmse(bmi_raw, bmi_gt),
        "MAE_BMI_from_hw":  mae(bmi_hw, bmi_gt),
        "RMSE_BMI_from_hw": rmse(bmi_hw, bmi_gt),
        "MAE_BMI_best": mae(bmi_best, bmi_gt),
        "RMSE_BMI_best": rmse(bmi_best, bmi_gt),
    }
    mpath = pred_path.replace("pred_", "eval_")
    join.to_csv(mpath, index=False)
    print(f" Eval ({Path(pred_path).name}):", metrics)
    return mpath, metrics

EVAL_PATHS = []
for p in ALL_PRED_PATHS:
    ep, _m = eval_and_save(p, gt)
    EVAL_PATHS.append(ep)

print("\nArchivos generados:")
for p in ALL_PRED_PATHS: print(" -", p)

for p in EVAL_PATHS:     print(" -", p)
