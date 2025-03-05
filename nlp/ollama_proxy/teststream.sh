curl -X POST http://107.170.79.167:5000/proxy/api/chat \
     -H "X-API-Key: c852148fa0f83063009c0b6c46e8bd2c65cfecba02076325c99f043eb6cf912c" \
     -H "Content-Type: application/json" \
     -d '{"model":"deepseek-r1:70b","options":{},"messages":[{"role":"user","content":"hello","images":[]}]}' \

