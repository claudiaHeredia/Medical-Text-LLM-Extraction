# ==== Inferencia y evaluación (4 prompts) con OpenVINO (Intel GPU Iris / CPU) ====
# - Diseñado para VS Code local
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

# ---------- Dependencias (instala sólo si faltan) ----------
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
        print("📦 Instalando:", need)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + need)
ensure(REQ)

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

# ------------ Rutas / Config ------------
NOTES_CSV = r"C:\Users\hered\Desktop\TFM\TFM\IMC2\eval_imc_fullnotes.csv"
GT_CSV    = r"C:\Users\hered\Desktop\TFM\TFM\IMC2\valid_imc.csv"
OUT_DIR   = str(Path("./outputs").resolve())


# CAMBIO DEL MODELO: Microsoft DialoGPT-large - Confiable y accesible
MODEL_ID  = os.getenv("MODEL_ID", "microsoft/DialoGPT-large")

# Inferencia - HIPERPARÁMETROS IDÉNTICOS
ATTEMPTS_PER_WIN = 4
N_WINDOWS_MAX    = 8
TEMP, TOP_P      = 0.4, 0.95
MAX_NEW          = 180

# Plausibilidad / tolerancias - IDÉNTICAS
H_MIN, H_MAX = 1.2, 2.2
W_MIN, W_MAX = 30, 300
BMI_MIN, BMI_MAX = 10, 80
BMI_TOL = 0.5
PROTOTYPES = {1.60, 1.70, 1.73, 1.75, 65.0, 70.0, 72.5, 75.0, 22.49, 24.2, 25.0}

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ------------ Carga de datos ------------
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
notes = notes.head(20)
print(f"Notas: {len(notes)} | GT: {len(gt)} | Intersección: {len(set(notes.patient_id)&set(gt.patient_id))}")

# ------------ Modelo OpenVINO (Intel GPU → CPU fallback) ------------
def get_ov_model_and_tokenizer(model_id: str, device_pref: str = "GPU"):
    print(f"\nCargando tokenizer: {model_id}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        print("✅ Tokenizer cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando tokenizer: {e}")
        return None, None, None

    device = "GPU" if device_pref.upper()=="GPU" else "CPU"
    print(f"Cargando modelo OpenVINO en {device}...")
    
    try:
        ov_model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,
            device=device,
            compile=True,
            trust_remote_code=False,  # DialoGPT no necesita trust_remote_code
            ov_config={"CACHE_DIR": str(Path(OUT_DIR)/"ov_cache")},
        )
        
        if getattr(ov_model.config, "pad_token_id", None) is None:
            ov_model.config.pad_token_id = tok.eos_token_id

        # Test de funcionamiento
        try:
            test_input = tok("Hello", return_tensors="pt")
            _ = ov_model.generate(test_input.input_ids, max_new_tokens=5)
            print(f"✅ Modelo operativo en {device}")
        except Exception as e:
            if device == "GPU":
                print(f"⚠️ Falló en GPU ({e}). Reintentando en CPU…")
                ov_model = OVModelForCausalLM.from_pretrained(
                    model_id, export=True, device="CPU", compile=True
                )
                print("✅ Modelo operativo en CPU")
            else:
                raise
                
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return None, None, None

    def llm_generate(prompt: str, max_new=MAX_NEW, temperature=TEMP, top_p=TOP_P, do_sample=True):
        try:
            # Para DialoGPT, usar un formato más simple
            formatted_prompt = f"System: {prompt}\nAssistant:"
            inputs = tok(formatted_prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            outputs = ov_model.generate(
                inputs.input_ids,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=float(temperature),
                top_p=float(top_p),
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )
            
            response = tok.decode(outputs[0], skip_special_tokens=True)
            # Extraer solo la respuesta del assistant
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            return response
        except Exception as e:
            print(f"⚠️ Error en generación: {e}")
            return ""

    def apply_chat_template(system_text: str, user_text: str):
        """Formato simple para DialoGPT"""
        return f"System: {system_text}\nUser: {user_text}\nAssistant:"

    return tok, llm_generate, apply_chat_template

print(f"Cargando modelo OpenVINO: {MODEL_ID}")
tokenizer, llm_generate, apply_chat_template = get_ov_model_and_tokenizer(MODEL_ID, device_pref="GPU")

if tokenizer is None:
    print("❌ No se pudo cargar el modelo. Saliendo...")
    exit(1)

# ------------ Ventanas / utils numéricas ------------
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

# ------------ 4 PROMPTS ------------
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

# ------------ Prompt builder / parser / scoring ------------
def build_prompt(system_text: str, window_text: str):
    # Para DialoGPT, formato más simple
    return f"{system_text}\n\nText: {window_text}\n\nRespond with JSON only:"

def safe_json(text: str, expect="triplet"):
    if not text: return None
    s = text.strip()
    
    # Limpiar texto - buscar JSON
    if "```json" in s:
        s = s.split("```json")[1].split("```")[0]
    elif "```" in s:
        s = s.split("```")[1]
    
    # Buscar entre llaves
    start = s.find('{')
    end = s.find('}')
    if start != -1 and end != -1 and end > start:
        json_str = s[start:end+1]
        # Limpiar el JSON
        json_str = json_str.replace("None", "null").replace("NaN", "null").replace(",}", "}")
        try:
            obj = json.loads(json_str)
            if not isinstance(obj, dict): return None
            if expect=="bmi_only":
                return obj if "bmi" in obj else None
            else:
                # Para DialoGPT, ser más permisivo con las keys
                has_height = any(key in obj for key in ["height_m", "height", "Height", "Height_m"])
                has_weight = any(key in obj for key in ["weight_kg", "weight", "Weight", "Weight_kg"]) 
                has_bmi = any(key in obj for key in ["bmi", "BMI", "Bmi"])
                if has_height or has_weight or has_bmi:
                    return obj
        except:
            pass
    return None

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

# --------- Ejecutores (simple/estricto/fewshot) ---------
def run_triplet(note_text: str, system_text: str):
    wins = [(s, c) for s, c in window_iter(note_text)]
    wins = sorted(wins, key=lambda x: int(not has_unit_token(x[1])))
    best={"h":None,"w":None,"b":None,"score":-1e9}
    
    for _, chunk in wins[:N_WINDOWS_MAX]:
        prompt = build_prompt(system_text, chunk)
        print(f"🔍 Procesando ventana con {len(chunk)} caracteres...")
        
        for attempt in range(ATTEMPTS_PER_WIN):
            raw = llm_generate(prompt)
            print(f"  Intento {attempt+1}: {raw[:100]}...")
            
            obj = safe_json(raw, expect="triplet")
            if obj is None:
                print("  ❌ No se pudo parsear JSON")
                continue
                
            # Mapear keys alternativas
            h = obj.get("height_m") or obj.get("height") or obj.get("Height") or obj.get("Height_m")
            w = obj.get("weight_kg") or obj.get("weight") or obj.get("Weight") or obj.get("Weight_kg") 
            b = obj.get("bmi") or obj.get("BMI") or obj.get("Bmi")
            
            try: h=float(h) if h else None
            except: h=None
            try: w=float(w) if w else None  
            except: w=None
            try: b=float(b) if b else None
            except: b=None
            
            h,w,b = clip_plausible(h,w,b)
            sc = score_triplet(h,w,b)
            print(f"  ✅ Extraído: h={h}, w={w}, b={b}, score={sc}")
            
            if sc>best["score"]:
                best={"h":h,"w":w,"b":b,"score":sc}
                
        if is_num(best["h"]) and is_num(best["w"]):
            break
            
    H = round(float(best["h"]),2) if is_num(best["h"]) else None
    W = round(float(best["w"]),1) if is_num(best["w"]) else None
    B_from = recompute_bmi(H,W) if (is_num(H) and is_num(W)) else None
    B = B_from if is_num(B_from) else (round(float(best["b"]),2) if is_num(best["b"]) else None)
    
    print(f"🎯 Resultado final: H={H}, W={W}, B_from={B_from}, B={B}")
    return H,W,B_from,B

def run_fewshot(note_text: str, system_text: str):
    wins = [(s, c) for s, c in window_iter(note_text)]
    wins = sorted(wins, key=lambda x: int(not has_unit_token(x[1])))
    best={"h":None,"w":None,"b":None,"score":-1e9}
    
    for _, chunk in wins[:N_WINDOWS_MAX]:
        # Para fewshot, construir prompt manualmente
        examples = "\n".join([f"Example {i+1}:\nInput: {ex_in}\nOutput: {ex_out}" 
                            for i, (ex_in, ex_out) in enumerate(FEW_SHOTS)])
        
        prompt = f"{system_text}\n\n{examples}\n\nNow extract from this text:\n{chunk}\n\nJSON output:"
        
        for attempt in range(ATTEMPTS_PER_WIN):
            raw = llm_generate(prompt)
            obj = safe_json(raw, expect="triplet")
            if obj is None: continue
            
            h = obj.get("height_m") or obj.get("height") 
            w = obj.get("weight_kg") or obj.get("weight")
            b = obj.get("bmi") or obj.get("BMI")
            
            try: h=float(h) if h else None
            except: h=None
            try: w=float(w) if w else None
            except: w=None
            try: b=float(b) if b else None
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

# --------- Encadenado (SPAN -> NORM) ---------
def chat_prompt(system, user):
    return apply_chat_template(system, user)

def run_chain_on_window(window_text: str):
    # Para DialoGPT, simplificar el encadenado
    SYS_EXTRACT = "Extract height, weight and BMI from the clinical text. Return JSON: {height_m, weight_kg, bmi}"
    
    raw = llm_generate(chat_prompt(SYS_EXTRACT, f"Text: {window_text}"))
    obj = safe_json(raw, expect="triplet") or {}
    
    return {
        "sentence": "",
        "height_m": obj.get("height_m") or obj.get("height"),
        "weight_kg": obj.get("weight_kg") or obj.get("weight"),
        "bmi": obj.get("bmi") or obj.get("BMI"),
        "bmi_source": "from_text",
        "check": "ok" if any([obj.get("height_m"), obj.get("weight_kg"), obj.get("bmi")]) else "insufficient"
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

# ------------ Orquestación (4 prompts) ------------
PROMPTS = {
    "v1_simple":    SYSTEM_SIMPLE,
    "v2_estricto":  SYSTEM_STRICT,
    "v3_fewshot":   SYSTEM_FEWSHOT,
    "v4_encadenado": None,   # chain
}

def save_rows(rows, out_csv):
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Guardado: {out_csv}")

ALL_PRED_PATHS = []

for pid_name, system_text in PROMPTS.items():
    out_csv = f"{OUT_DIR}/pred_{pid_name}_{MODEL_ID.split('/')[-1]}_n{len(notes)}.csv"
    rows=[]
    print(f"\n🎯 Procesando estrategia: {pid_name}")
    
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

# ------------ Evaluación contra GT ------------
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

    # BMI "mejor" del modelo
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
    print(f"📊 Eval ({Path(pred_path).name}):", metrics)
    return mpath, metrics

EVAL_PATHS = []
for p in ALL_PRED_PATHS:
    ep, _m = eval_and_save(p, gt)
    EVAL_PATHS.append(ep)

print("\nArchivos generados:")
for p in ALL_PRED_PATHS: print(" -", p)
for p in EVAL_PATHS:     print(" -", p)