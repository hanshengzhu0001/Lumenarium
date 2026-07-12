#!/bin/bash
cd /ssd/kevinzyz/imaginarium/Imaginarium-repo
while true; do
    gem_ok=$(grep -c "✅" /mnt/kevinzyz/artifacts/Imaginarium-repo/batch_logs/synmap_paper30_gemini.out 2>/dev/null | tr -d '\n'); gem_ok=${gem_ok:-0}
    gem_fail=$(grep -c "❌" /mnt/kevinzyz/artifacts/Imaginarium-repo/batch_logs/synmap_paper30_gemini.out 2>/dev/null | tr -d '\n'); gem_fail=${gem_fail:-0}
    gpt_ok=$(grep -c "✅" /mnt/kevinzyz/artifacts/Imaginarium-repo/batch_logs/synmap_paper30_gpt54.out 2>/dev/null | tr -d '\n'); gpt_ok=${gpt_ok:-0}
    gpt_fail=$(grep -c "❌" /mnt/kevinzyz/artifacts/Imaginarium-repo/batch_logs/synmap_paper30_gpt54.out 2>/dev/null | tr -d '\n'); gpt_fail=${gpt_fail:-0}
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head -4 | tr '\n' ' ')
    echo "[$(date +%H:%M:%S)] Gemini:✅${gem_ok}❌${gem_fail} | GPT:✅${gpt_ok}❌${gpt_fail} | GPU:${gpu}"
    g_tot=$((gem_ok + gem_fail))
    p_tot=$((gpt_ok + gpt_fail))
    if [ "$g_tot" -ge 30 ] && [ "$p_tot" -ge 30 ]; then
        echo "[$(date +%H:%M:%S)] 🎉 BOTH V3 COMPLETE"
        break
    fi
    sleep 120
done
