# MANGO1.5-Qwen3.5-9B on DGX Spark with vLLM

คู่มือเตรียม environment ดาวน์โหลดโมเดลจาก Hugging Face และรัน vLLM ด้วย Docker
Compose สำหรับโปรไฟล์ E0/E1

-   **E0:** BF16 autoregressive inference
-   **E1:** BF16 + checkpoint MTP (speculative decoding)
-   รัน **ครั้งละหนึ่งโปรไฟล์เท่านั้น** เพราะทั้งสองใช้ host network และพอร์ต `8001`
    เหมือนกัน

## 1. ตรวจสอบเครื่องมือและ GPU

``` bash
uname -m
docker --version
docker compose version
docker info
nvidia-smi
```

ผลของ `uname -m` บน ARM64 คือ `aarch64` ส่วน AMD64 มักเป็น `x86_64` การตรวจ
GPU ให้ใช้เครื่องมือที่รองรับกับ driver stack ของระบบ DGX Spark ที่ติดตั้งไว้

## 2. สร้าง Python virtual environment สำหรับ Hugging Face CLI

``` bash
mkdir -p ~/mango-vllm
cd ~/mango-vllm

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade huggingface_hub

hf --help
hf download --help
```

ใช้ virtual environment แยกจาก Python ของระบบ เพื่อไม่ให้การติดตั้ง CLI
กระทบแพ็กเกจอื่น

## 3. เข้าสู่ระบบ Hugging Face (ถ้าจำเป็น)

หาก repository เป็น gated/private หรือบัญชีต้องได้รับสิทธิ์ ให้เข้าสู่ระบบด้วย token
ที่มีสิทธิ์:

``` bash
hf auth login
```

อย่า commit token ลง Git หากโมเดลเป็น public และดาวน์โหลดได้โดยไม่ต้องยืนยันตัวตน
สามารถข้ามขั้นตอนนี้ได้

## 4. ดาวน์โหลดโมเดลลง local directory

``` bash
mkdir -p models

hf download CMKL/MANGO1.5-Qwen3.5-9B \
  --local-dir ./models/MANGO1.5-Qwen3.5-9B

ls -lah ./models/MANGO1.5-Qwen3.5-9B
```

ตรวจว่าไฟล์ weights, configuration และ tokenizer ที่ repository
กำหนดดาวน์โหลดครบก่อนเริ่ม container

> **หมายเหตุเกี่ยวกับ E1/MTP:** Compose ที่ให้มาชี้ E0 และ E1 ไปยัง directory
> เดียวกัน (`./models/MANGO1.5-Qwen3.5-9B`) แต่ E1 เปิด
> `--speculative-config` สำหรับ MTP ด้วย ต้องตรวจสอบว่า directory/checkpoint
> ที่ดาวน์โหลดมีข้อมูล MTP ที่ vLLM คาดหวังจริง การเพิ่ม `--speculative-config`
> เพียงอย่างเดียวไม่ได้ดาวน์โหลดหรือสร้าง checkpoint MTP ให้อัตโนมัติ หาก checkpoint
> MTP แยกต่างหาก ให้ทำตามตำแหน่งและรูปแบบที่ผู้เผยแพร่โมเดลกำหนด

## 5. สร้าง `compose.yaml`

สร้างไฟล์ `compose.yaml` ที่ root ของโปรเจกต์ แล้วใส่ configuration สำหรับ E0/E1
ดังนี้

``` yaml
x-vllm-common: &vllm-common
  image: vllm/vllm-openai:latest
  # DGX Spark / ARM64: ใช้ linux/arm64 เมื่อ image รองรับจริง
  platform: linux/arm64
  network_mode: host
  ipc: host
  shm_size: 16gb
  ulimits:
    memlock: -1
    stack: 67108864
  runtime: nvidia
  volumes:
    - ./models/MANGO1.5-Qwen3.5-9B:/models/mango:ro
  environment:
    HF_HUB_OFFLINE: "1"
    TRANSFORMERS_OFFLINE: "1"
    VLLM_LOGGING_LEVEL: INFO
  restart: "no"

services:
  mango-e0:
    <<: *vllm-common
    profiles: ["e0"]
    container_name: mango-e0
    command:
      - /models/mango
      - --host
      - 0.0.0.0
      - --port
      - "8001"
      - --served-model-name
      - CMKL/MANGO1.5-Qwen3.5-9B
      - --tensor-parallel-size
      - "1"
      - --dtype
      - bfloat16
      - --max-model-len
      - "8192"
      - --max-num-seqs
      - "128"
      - --max-num-batched-tokens
      - "8192"
      - --gpu-memory-utilization
      - "0.70"
      - --reasoning-parser
      - qwen3
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_coder

  mango-e1:
    <<: *vllm-common
    profiles: ["e1"]
    container_name: mango-e1
    command:
      - /models/mango
      - --trust-remote-code
      - --language-model-only
      - --host
      - 0.0.0.0
      - --port
      - "8001"
      - --served-model-name
      - CMKL/MANGO1.5-Qwen3.5-9B
      - --tensor-parallel-size
      - "1"
      - --dtype
      - bfloat16
      - --max-model-len
      - "40960"
      - --max-num-seqs
      - "128"
      - --max-num-batched-tokens
      - "8192"
      - --gpu-memory-utilization
      - "0.90"
      - --enable-prefix-caching
      - --reasoning-parser
      - qwen3
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_coder
      - --speculative-config
      - '{"method":"mtp","num_speculative_tokens":1}'
      - --enable-chunked-prefill
      - --kv-cache-dtype
      - fp8
      - --async-scheduling
```

### ความหมายของค่าหลัก

  -----------------------------------------------------------------------------------------
  Setting                                        E0                   E1 ความหมาย
  ---------------------------- -------------------- -------------------- ------------------
  `--dtype`                              `bfloat16`           `bfloat16` dtype
                                                                         สำหรับการคำนวณหลัก

  `--max-model-len`                          `8192`              `40960` ความยาว context
                                                                         สูงสุดที่ตั้งไว้

  `--max-num-seqs`                            `128`                `128` ขีดจำกัดจำนวน
                                                                         sequence
                                                                         ที่ประมวลผลต่อ
                                                                         iteration ตามกลไก
                                                                         vLLM

  `--max-num-batched-tokens`                 `8192`               `8192` งบ token สำหรับจัด
                                                                         batch ต่อ iteration

  `--gpu-memory-utilization`                 `0.70`               `0.90` สัดส่วน GPU memory ที่
                                                                         vLLM ตั้งเป้าใช้

  Prefix caching                                 ปิด                  เปิด reuse KV blocks
                                                                         ของ prefix ที่ตรงกัน

  Speculative decoding                           ปิด         MTP, 1 token เปิดใช้ MTP สำหรับ
                                                                         speculative
                                                                         decoding

  KV cache dtype                            ค่าเริ่มต้น                `fp8` ใช้ FP8 สำหรับ KV
                                                                         cache ใน E1

  Chunked prefill                           ไม่ได้ระบุ                  เปิด อนุญาตให้แบ่ง prefill
                                                                         เป็น chunks

  Async scheduling                          ไม่ได้ระบุ                  เปิด เปิด asynchronous
                                                                         scheduling
  -----------------------------------------------------------------------------------------

ค่าข้างต้นคงตามโปรไฟล์ที่ให้มา ไม่ใช่การรับประกันว่าเหมาะที่สุดกับทุก workload
ตรวจสอบความเข้ากันได้ของ flags กับ image/version ที่ใช้ โดยเฉพาะ
`--language-model-only`, MTP และ `--async-scheduling`

## 6. ตรวจสอบ Compose และ image architecture

``` bash
docker compose config
docker compose --profile e0 config --services
docker compose --profile e1 config --services
```

ตรวจ architecture ของ image ที่ pull มาแล้ว:

``` bash
docker pull vllm/vllm-openai:latest
docker image inspect vllm/vllm-openai:latest \
  --format '{{.Os}}/{{.Architecture}}'
```

การตรวจ image architecture ไม่ได้รับประกันว่า image รองรับทุก feature หรือ GPU
runtime ของเครื่อง ให้ตรวจ log และทดสอบการเริ่ม container จริงด้วย

## 7. รัน E0 หรือ E1 (ครั้งละหนึ่งโปรไฟล์)

### E0 --- BF16 autoregressive

``` bash
docker compose --profile e0 up -d
docker logs -f mango-e0
```

หยุด E0 ก่อนเปลี่ยนไป E1:

``` bash
docker compose --profile e0 down
docker ps
```

### E1 --- BF16 + MTP

``` bash
docker compose --profile e1 up -d
docker logs -f mango-e1
```

หยุด E1:

``` bash
docker compose --profile e1 down
```

อย่ารัน E0 และ E1 พร้อมกันตามการออกแบบนี้ เพราะใช้ `network_mode: host` และพอร์ต
`8001` เดียวกัน และอาจแข่งขันใช้ GPU memory

## 8. ตรวจสถานะและทดสอบ API

ดู log:

``` bash
docker logs -f mango-e0
# หรือ
docker logs -f mango-e1
```

ตรวจ health endpoint (ถ้า image/version ที่ใช้รองรับ):

``` bash
curl -sS http://localhost:8001/health
```

ตรวจ metrics:

``` bash
curl -sS http://localhost:8001/metrics
```

ทดสอบ Chat Completions API:

``` bash
curl -sS http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "CMKL/MANGO1.5-Qwen3.5-9B",
    "messages": [
      {"role": "user", "content": "Explain speculative decoding in two sentences."}
    ],
    "temperature": 0
  }'
```

ตรวจ `choices`, `message.content`, `finish_reason` และข้อมูล usage ที่ API
ส่งกลับมา

## 9. Benchmark: Cold cache กับ Warm cache

E1 เปิด `--enable-prefix-caching` ดังนั้น request ที่ใช้ prefix ซ้ำอาจ reuse KV
blocks ที่ยังอยู่ใน cache การรัน benchmark รอบถัดไปโดยไม่รีสตาร์ทเซิร์ฟเวอร์จึงอาจให้
latency/throughput ต่างจากรอบแรก

เพื่อเปรียบเทียบ config อย่างยุติธรรม:

1.  กำหนดว่าจะวัด **cold cache** หรือ **warm cache**
2.  ใช้ dataset, ลำดับ request, concurrency และ generation parameters เดิม
3.  เก็บ Score, Accuracy, SLO, TTFT, TPOT, error count และ Prefix Cache
    hit rate
4.  สำหรับ cold-cache comparison ให้เริ่มจาก cache ว่างทุกครั้ง โดยรีสตาร์ท vLLM
    ก่อนแต่ละรอบหรือใช้วิธี reset cache ที่รองรับในเวอร์ชันนั้น
5.  สำหรับ warm-cache benchmark ให้ใช้ขั้นตอน warm-up
    เดียวกันทุกรอบและรายงานเงื่อนไขนี้ไว้ในผลทดลอง

อย่าเปรียบเทียบ cold-cache run กับ warm-cache run
โดยไม่ระบุความแตกต่างของเงื่อนไข

## 10. Troubleshooting

### ดาวน์โหลดโมเดลไม่สำเร็จ

-   ตรวจชื่อ repository และสิทธิ์ Hugging Face
-   ตรวจพื้นที่ดิสก์และการเชื่อมต่อ
-   หากเป็น gated/private repository ให้ยืนยันตัวตนก่อนดาวน์โหลด
-   ตรวจว่าไฟล์ใน local directory ครบก่อนเปิดโหมด offline

### Container เริ่มไม่ได้หรือไม่พบ GPU

-   ตรวจ Docker และ NVIDIA runtime configuration
-   ตรวจ architecture ของ host กับ image
-   อ่าน log ด้วย `docker logs mango-e0` หรือ `docker logs mango-e1`
-   ยืนยันว่า image/tag รองรับ platform และ feature ที่ต้องใช้

### Out of memory

-   ตรวจ GPU memory และ log
-   ทดลองลด `--max-model-len` หรือ `--max-num-seqs` ทีละตัวแปร
-   `--gpu-memory-utilization` ที่สูงขึ้นไม่ได้แปลว่าจะเร็วขึ้นเสมอ
    และอาจเพิ่มความเสี่ยง OOM

### E1 ใช้ MTP ไม่ได้

-   ยืนยันว่า checkpoint MTP มีอยู่และตรงกับโมเดลหลัก
-   ตรวจ log ตอนเริ่ม server และความเข้ากันได้กับ vLLM version
-   อย่าสรุปว่า MTP ทำงานแล้วเพียงเพราะ command มี `--speculative-config`

### Port 8001 ถูกใช้งานแล้ว

-   หยุดโปรไฟล์ที่กำลังทำงานก่อน
-   ตรวจ container ด้วย `docker ps`
-   ตรวจ process ที่ครอบครองพอร์ตด้วยเครื่องมือของระบบ

## 11. คำสั่งเริ่มต้นแบบย่อ

``` bash
# สร้าง environment และติดตั้ง Hugging Face CLI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip huggingface_hub

# ดาวน์โหลดโมเดล
mkdir -p models
hf download CMKL/MANGO1.5-Qwen3.5-9B \
  --local-dir ./models/MANGO1.5-Qwen3.5-9B

# ตรวจ Compose
docker compose config

# รัน E0
docker compose --profile e0 up -d
docker logs -f mango-e0

# หยุด E0 ก่อนรัน E1
docker compose --profile e0 down

# รัน E1
docker compose --profile e1 up -d
docker logs -f mango-e1
```

## หมายเหตุเกี่ยวกับโปรไฟล์อื่นใน Compose ต้นฉบับ

คู่มือนี้ครอบคลุมเฉพาะ E0/E1 ตามคำอธิบายส่วนต้น โปรไฟล์ E2/E2-tools/E3 เป็น
workflow แยกต่างหาก ต้องตรวจ model paths, checkpoint paths, image versions
และ dependencies ของแต่ละโปรไฟล์ก่อนใช้งาน
