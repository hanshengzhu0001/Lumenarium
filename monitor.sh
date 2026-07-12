#!/bin/bash
# 每 30 分钟监控 batch 评估进度
LOG=/ssd/kevinzyz/imaginarium/Imaginarium-repo/monitor.log
DIR=/ssd/kevinzyz/imaginarium/Imaginarium-repo

while true; do
    cd $DIR
    total=$(ls saved_results/*_result/S4_layout_refinement/*_placement_info_s4.json 2>/dev/null | wc -l)
    v1=$(ls saved_results/*_v1_result/S4_layout_refinement/*_placement_info_s4.json 2>/dev/null | wc -l)
    v3=$(ls saved_results/*_v3_result/S4_layout_refinement/*_placement_info_s4.json 2>/dev/null | wc -l)
    active=$(pgrep -f 'run_imaginarium' 2>/dev/null | wc -l)
    failed=$(grep -l 'PIPELINE FAILED' saved_results/*_result/pipeline.log 2>/dev/null | wc -l)
    latest=$(ls -t saved_results/*_result/S4_layout_refinement/*_placement_info_s4.json 2>/dev/null | head -3 | xargs -I{} dirname {} | xargs -I{} dirname {} | xargs -I{} basename {} | sed 's/_result//' | tr '\n' ' ')

    echo "[$(date '+%m-%d %H:%M')] total:$total/302 v1:$v1 v3:$v3 active:$active fail:$failed latest:$latest" | tee -a $LOG

    if [ "$total" -ge 302 ]; then
        echo "[$(date)] ALL DONE!" | tee -a $LOG
        break
    fi
    sleep 1800  # 30 分钟
done
