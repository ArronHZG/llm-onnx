SGLANG_USE_MODELSCOPE=true \
nohup python \
-m sglang.launch_server \
--model-path /home/hadoop-djst-algoplat/model/Qwen/Qwen3.6-27B \
--host 0.0.0.0 \
--port 44400 \
--tp-size 8 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--speculative-algo EAGLE3 \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--enable-metrics \
--load-format auto \
> sglang.log &
>

nohup python \
-m sglang.launch_server \
--model-path /home/hadoop-djst-algoplat/model/Qwen/Qwen3.5-35B-A3B \
--host 0.0.0.0 \
--port 44400 \
--tp-size 8 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--speculative-algo NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--enable-metrics \
--load-format auto \
> sglang.log &

> 
> 

编辑本地 ~/.openclaw/openclaw.json

openclaw config get agents.defaults.model.primary

执行 openclaw doctor

ray stop && ray start --head --dashboard-host=33.29.18.248 --dashboard-port=8420

# Master node environment variables
export GF_SERVER_HTTP_PORT=44395                     # Grafana service default port (customizable)
export PROMETHEUS_PORT=44398                         # Prometheus service default port (customizable)
export RAY_HEAD_PORT=6379                           # Ray master node port (customizable)
export RAY_DASHBOARD_PORT=8420                      # Ray dashboard default port (customizable)
export GRAFANA_PATHS_DATA=/tmp/grafana              # Grafana data storage directory (customizable)
export master_ip=33.29.18.248
export RAY_GRAFANA_HOST="http://${master_ip}:${GF_SERVER_HTTP_PORT}"        # Ray-associated Grafana address
export RAY_PROMETHEUS_HOST="http://${master_ip}:${PROMETHEUS_PORT}"         # Ray-associated Prometheus address


nohup ~/grafana/bin/grafana-server \
--config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
--homepath ~/grafana \
web > grafana.log 2>&1 &


nohup /prometheus/prometheus \
--config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml \
--web.enable-lifecycle \
--web.listen-address=0.0.0.0:44398 \
> prometheus.log 2>&1 &


/tmp/ray/session_latest/metrics/prometheus/prometheus.yml
# 编辑 Prometheus 配置文件
scrape_configs:
- job_name: "rollout"
  static_configs:
    - targets: ["33.29.18.248:44400"]  
    - 
curl -X POST http://10.148.11.32:44398/-/reload