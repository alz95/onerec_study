from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

LOG_PATH = Path("runs") / "metrics.jsonl"


INDEX_HTML = b"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>OneRec Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
      body { font-family: -apple-system, system-ui, Arial; margin: 16px; }
      .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
      canvas { background: #fff; border: 1px solid #eee; border-radius: 8px; padding: 8px; }
      .row { margin-bottom: 16px; }
      .pill { display: inline-block; padding: 4px 8px; border-radius: 12px; background: #f4f4f4; margin-right: 8px; }
    </style>
  </head>
  <body>
    <h2>OneRec Training Dashboard</h2>
    <div class="row">
      <span class="pill" id="device"></span>
      <span class="pill" id="batch"></span>
      <span class="pill" id="data"></span>
      <span class="pill" id="sample"></span>
    </div>
    <div class="grid">
      <canvas id="lossChart"></canvas>
      <canvas id="timeChart"></canvas>
      <canvas id="recallChart"></canvas>
      <canvas id="ndcgChart"></canvas>
    </div>
    <script>
      const lossCtx = document.getElementById('lossChart').getContext('2d');
      const timeCtx = document.getElementById('timeChart').getContext('2d');
      const recallCtx = document.getElementById('recallChart').getContext('2d');
      const ndcgCtx = document.getElementById('ndcgChart').getContext('2d');

      const mkChart = (ctx, label, yLabel) => new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [{ label, data: [], borderColor: '#1f77b4', tension: 0.2 }] },
        options: { responsive: true, scales: { y: { title: { display: true, text: yLabel } }, x: { title: { display: true, text: 'epoch' } } } }
      });

      const lossChart = mkChart(lossCtx, 'loss', 'loss');
      const timeChart = mkChart(timeCtx, 'time_s', 'seconds');
      const recallChart = mkChart(recallCtx, 'Recall@K', 'recall');
      const ndcgChart = mkChart(ndcgCtx, 'NDCG@K', 'ndcg');

      async function refresh() {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        const epochs = data.map(d => d.epoch);
        lossChart.data.labels = epochs;
        lossChart.data.datasets[0].data = data.map(d => d.loss);
        timeChart.data.labels = epochs;
        timeChart.data.datasets[0].data = data.map(d => d.time_s);
        const kRecall = data.length ? Object.keys(data[0].metrics).filter(k => k.startsWith('Recall@')) : [];
        const kNdcg = data.length ? Object.keys(data[0].metrics).filter(k => k.startsWith('NDCG@')) : [];
        recallChart.data.labels = epochs;
        ndcgChart.data.labels = epochs;
        recallChart.data.datasets = kRecall.map((k, i) => ({ label: k, data: data.map(d => d.metrics[k]), borderColor: ['#1f77b4','#ff7f0e','#2ca02c','#d62728'][i%4], tension: 0.2 }));
        ndcgChart.data.datasets = kNdcg.map((k, i) => ({ label: k, data: data.map(d => d.metrics[k]), borderColor: ['#9467bd','#8c564b','#e377c2','#7f7f7f'][i%4], tension: 0.2 }));
        lossChart.update(); timeChart.update(); recallChart.update(); ndcgChart.update();
        if (data.length) {
          const last = data[data.length - 1];
          document.getElementById('device').innerText = 'device: ' + (last.device || 'n/a');
          document.getElementById('batch').innerText = 'batch: ' + (last.batch_size || 'n/a');
          document.getElementById('data').innerText = 'data: ' + (last.data_source || 'n/a');
          document.getElementById('sample').innerText = 'mode: ' + (last.sample_mode || 'n/a');
        }
      }
      setInterval(refresh, 2000);
      refresh();
    </script>
  </body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/" or p == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML)
            return
        if p == "/api/metrics":
            items = []
            if LOG_PATH.exists():
                with LOG_PATH.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            continue
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(items).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()


def run():
    addr = ("0.0.0.0", 8765)
    httpd = HTTPServer(addr, Handler)
    print("Serving dashboard at http://localhost:8765/")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
