# vTTS — Virtual Text-to-Speech

**Continuous batching para modelos TTS. Como vLLM, pero para voz.**

[English documentation](README.md)

vTTS permite servir multiples peticiones de texto-a-voz simultaneamente en una sola GPU usando continuous batching con join/leave dinamico — la misma tecnica que vLLM usa para LLMs, aplicada a generacion de voz.

## Resultados

| Metrica | Valor |
|---|---|
| Voces simultaneas en 1 GPU | **8-10** (RTX 3060 12GB) |
| Tiempo hasta primer audio (1 usuario) | **192ms** (RTX 3090 Ti) / **254ms** (RTX 3060) |
| Tiempo hasta primer audio (4 usuarios) | **254ms** (RTX 3090 Ti) / **382ms** (RTX 3060) |
| Tiempo hasta primer audio (8 usuarios) | **368ms** (RTX 3090 Ti) / **570ms** (RTX 3060) |
| VRAM extra por usuario | **~2.5MB** (solo KV cache) |
| Throughput (5 simultaneos) | **3.6 segundos de audio/segundo real** |

Medido con Qwen3-TTS-12Hz-1.7B-CustomVoice (3.4GB VRAM).

## Como Funciona

Los servidores TTS tradicionales procesan una peticion a la vez — cada usuario bloquea la GPU durante toda la generacion. vTTS cambia esto ejecutando todas las peticiones activas en un **unico forward pass batched** por step, con peticiones entrando y saliendo del batch dinamicamente.

### Modos de Arquitectura

#### M1 — Batch por Epocas (simple)

Todas las peticiones de un batch empiezan y terminan juntas. Las nuevas esperan a que termine el batch actual. Simple y eficiente, pero agrega latencia para nuevas llegadas.

Ideal para: **doblaje por lotes**, procesamiento offline, escenarios donde todos los textos se conocen de antemano.

```
Batch 1: [req_A, req_B, req_C] → generan juntas → terminan todas
         (nuevas peticiones esperan aqui)
Batch 2: [req_D, req_E] → generan juntas → terminan todas
```

#### M2 — Continuous Batching (avanzado)

Las peticiones entran y salen del loop de generacion en cualquier momento. Una nueva peticion entra al batch en un step (~80ms), sin esperar a que las demas terminen. Este es el **equivalente a vLLM** para TTS.

Ideal para: **agentes de voz en tiempo real**, APIs de streaming, cualquier escenario con peticiones impredecibles.

```
Step 1:  [req_A, req_B]           ← A y B generando
Step 10: [req_A, req_B, req_C]    ← C entra (dynamic join)
Step 25: [req_A, req_C]           ← B termino y salio (dynamic leave)
Step 30: [req_A, req_C, req_D]    ← D entra
Step 40: [req_C, req_D]           ← A termino
```

**Como funciona el forward batched internamente:**

1. Cada slot tiene su propio KV cache (distintas longitudes) y generation_step
2. Los KV caches se paddean a la misma longitud con attention_mask=0
3. El condicionamiento de texto se aplica per-slot antes de batchear
4. Un solo forward pass del transformer procesa todos los slots
5. Despues del forward, los KV caches se extraen per-slot (quitando padding)

## Modelos Soportados

| Modelo | Parametros | VRAM | Modo | Voces |
|---|---|---|---|---|
| [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 0.6B | ~1.2GB | Clonacion de voz | Cualquiera (desde audio de referencia) |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 1.7B | ~3.4GB | Clonacion de voz | Cualquiera (desde audio de referencia) |
| [Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 0.6B | ~1.2GB | Voces integradas | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |
| [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 1.7B | ~3.4GB | Voces integradas | aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian |

Los modelos se descargan automaticamente de HuggingFace en el primer uso. Si ya tienes un modelo descargado, puedes apuntar a la ruta local en la configuracion (ver [Configuracion](#configuracion)).

## Requisitos

- Python >= 3.10
- GPU CUDA (probado en RTX 3060 12GB, RTX 3090 Ti 24GB)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — libreria de inferencia de Qwen3-TTS

## Instalacion

```bash
git clone https://github.com/caimari/vtts.git
cd vtts
pip install -e .
```

Tambien necesitas la libreria de inferencia de Qwen3-TTS:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
pip install -e Qwen3-TTS/
```

## Inicio Rapido

### Voces Integradas (modelos CustomVoice)

```python
from vtts import ContinuousWorker, TTSRequest
import numpy as np
import soundfile as sf

# Cargar modelo (se descarga automaticamente de HuggingFace)
worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
    emit_every_frames=6,
)
worker.load_model()

# Generar audio
req = TTSRequest(
    request_id="hola",
    text="Hola, esto es una prueba del sistema vTTS.",
    voice_id="vivian",
    language="Spanish",
)
worker.prefill_slot_speaker(req)

while worker.num_active_slots > 0:
    worker.step()

# Recoger el audio
chunks = []
while not req.chunk_queue.empty():
    chunk = req.chunk_queue.get_nowait()
    if chunk is not None:
        chunks.append(chunk)

audio = np.concatenate(chunks)
sf.write("salida.wav", audio, req.sample_rate)
```

### Clonacion de Voz (modelos Base)

```python
from vtts import ContinuousWorker, TTSRequest, VoiceRegistry

worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda:0",
)
worker.load_model()

# Registrar una voz desde un audio de referencia.
# El audio debe durar ~5-10 segundos de habla limpia.
# ref_text debe ser la transcripcion EXACTA de lo que se dice en el audio.
registry = VoiceRegistry()
registry.register(
    model=worker.model,
    ref_audio_path="./voices/mi_voz.wav",
    ref_text="Las palabras exactas que se dicen en el audio de referencia.",
    voice_id="mi_voz",
)

req = TTSRequest(
    request_id="test_clon",
    text="Esto sonara como la voz de referencia.",
    voice_id="mi_voz",
    language="Spanish",
)
worker.prefill_slot(req, registry)

while worker.num_active_slots > 0:
    worker.step()
```

### Multiples Voces Simultaneas

```python
# Todas estas se generan en UN SOLO forward pass batched por step
requests = []
for speaker in ["vivian", "ryan", "serena", "eric"]:
    req = TTSRequest(
        request_id=speaker,
        text=f"Hola desde {speaker}!",
        voice_id=speaker,
        language="Spanish",
    )
    worker.prefill_slot_speaker(req)
    requests.append(req)

# Un solo loop impulsa las 4 generaciones simultaneamente
while worker.num_active_slots > 0:
    worker.step()
```

## Clonacion de Voz: Guia del Audio de Referencia

Para modelos Base (clonacion de voz), necesitas un audio de referencia:

- **Duracion**: 5-10 segundos (mejor corto que demasiado largo)
- **Contenido**: Habla clara, minimo ruido de fondo
- **Formato**: WAV (mono o estereo, cualquier frecuencia de muestreo)
- **ref_text**: Debe ser la **transcripcion exacta** de lo que se dice en el audio. Si la transcripcion no coincide con el audio, la calidad se degradara significativamente
- **Idioma**: El idioma del audio de referencia no necesita coincidir con el idioma objetivo

## Configuracion

Copia `config.example.yaml` a `config.yaml`:

```bash
cp config.example.yaml config.yaml
```

Configuracion clave:

```yaml
model:
  name: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  device: "cuda:0"

  # Si ya descargaste el modelo, apunta aqui para evitar
  # volver a descargarlo:
  # local_path: "/ruta/a/tu/modelo/descargado"

worker:
  emit_every_frames: 6    # Menor = menor latencia
  max_slots: 16           # Max generaciones simultaneas
  max_frames: 3600        # Max duracion audio (12 frames = 1 segundo)
```

### Usar un Modelo Ya Descargado

Si ya tienes un modelo Qwen3-TTS en disco, configura `local_path`:

```yaml
model:
  name: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  local_path: "/home/user/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
```

O pasalo directamente en codigo:

```python
worker = ContinuousWorker(
    model_name="/home/user/models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
)
```

## Servidor API

```bash
python -m vtts.server --config config.yaml
```

### Endpoints

| Metodo | Ruta | Descripcion |
|---|---|---|
| POST | `/v1/tts/generate` | Generar audio (stream SSE) |
| POST | `/v1/voices/register` | Registrar voz clonada |
| GET | `/v1/voices` | Listar voces disponibles |
| GET | `/v1/stats` | Estadisticas del servidor |
| GET | `/health` | Health check |

## Ajuste de Latencia

El parametro `emit_every_frames` controla el balance entre latencia y eficiencia:

| emit_every_frames | Tamano chunk | TTFB (1 usuario) | Ideal para |
|---|---|---|---|
| 3 | ~250ms | ~200ms | Agentes de voz, tiempo real |
| 6 | ~500ms | ~400ms | APIs de streaming |
| 20 | ~1.7s | ~1.4s | Procesamiento por lotes |

Para agentes de voz, usa `emit_every_frames=3` para TTFB sub-300ms.

## Presupuesto de VRAM

| Componente | VRAM |
|---|---|
| Modelo 0.6B (bf16) | ~1.2 GB |
| Modelo 1.7B (bf16) | ~3.4 GB |
| Vocoder | ~0.8 GB |
| KV cache por usuario (~200 frames) | ~2.5 MB |
| Overhead CUDA | ~0.5 GB |

Con 10 usuarios simultaneos y modelo 1.7B: ~4.7 GB total. Cabe sin problemas en una GPU de 12GB.

## Uso de vTTS en un Pipeline de Agentes de Voz

vTTS esta disenado para usarse como componente TTS en un pipeline de agente de voz en tiempo real. Una arquitectura tipica seria:

```
Usuario (telefono/navegador)
    │
    ▼
Gateway SIP / WebRTC
    │
    ▼
STT (Speech-to-Text)  ──►  Texto transcrito
    │
    ▼
LLM (Modelo de Lenguaje) ──►  Texto de respuesta
    │
    ▼
vTTS (esta libreria)   ──►  Chunks de audio (streaming)
    │
    ▼
El usuario escucha la respuesta
```

Tu orquestador conecta estos componentes. vTTS se integra como libreria — no necesita un servidor separado:

```python
from vtts import ContinuousWorker, TTSRequest

# Inicializar una vez al arrancar
worker = ContinuousWorker(
    model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device="cuda:0",
    emit_every_frames=3,  # Baja latencia para agentes de voz
)
worker.load_model()

# En el handler de tu orquestador:
async def handle_call(user_id, llm_response_text):
    req = TTSRequest(
        request_id=user_id,
        text=llm_response_text,
        voice_id="vivian",
        language="Spanish",
    )
    worker.prefill_slot_speaker(req)

    # Enviar chunks de audio al usuario conforme se generan
    while True:
        chunk = await req.chunk_queue.get()
        if chunk is None:
            break
        send_audio_to_user(user_id, chunk, sample_rate=req.sample_rate)
```

La ventaja clave de vTTS es que **multiples llamadas generan simultaneamente** en un unico forward pass batched. Si 10 usuarios estan en llamadas al mismo tiempo, la GPU procesa los 10 en cada step — sin colas, sin esperas.

Para el lado STT, la mayoria de motores (Whisper, Deepgram, etc.) ya soportan peticiones concurrentes de forma nativa y no requieren batching personalizado.

## Por Que No Simplemente Usar Multiples Procesos?

Se podrian ejecutar N copias del modelo (una por usuario), pero:

| Enfoque | Usuarios | VRAM | Velocidad |
|---|---|---|---|
| 3 procesos separados | 3 | 7.3 GB cada uno (22 GB total) | 43s por generacion |
| vTTS continuous batching | 10 | 3.4 GB total (+2.5 MB/usuario) | 3.6 seg audio/seg real |

Multiples procesos duplican los pesos del modelo en VRAM. vTTS comparte un unico modelo entre todos los usuarios, batching sus forward passes juntos. En una GPU de 12GB, multiproceso llega a maximo 3 usuarios. vTTS sirve 10+ sin problemas.

## Autor

Creado por [Antoni Caimari Caldes](https://github.com/caimari) — acaimari22@gmail.com

## Licencia

MIT — ver [LICENSE](LICENSE).
