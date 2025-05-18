from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json

from eigen_koan_matrix import create_random_ekm, EigenKoanMatrix

app = FastAPI(title="EKM Explorer")

# In-memory matrix state
default_size = 4
matrix: EigenKoanMatrix = create_random_ekm(default_size)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>EKM Explorer</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
#matrix td { border: 1px solid #666; padding: 6px; cursor: pointer; transition: background-color 0.3s; }
#matrix td.selected { background-color: #e0f7fa; }
.fade { opacity: 0; transition: opacity 0.5s; }
</style>
</head>
<body>
<h1>Eigen-Koan Matrix Explorer</h1>
<button id='random'>Randomize Matrix</button>
<div id='stats'></div>
<table id='matrix'></table>
<script>
let currentMatrix = null;
let currentPath = [];
async function loadMatrix(){
  const res = await fetch('/api/matrix');
  currentMatrix = await res.json();
  const table = document.getElementById('matrix');
  table.classList.add('fade');
  table.innerHTML='';
  for(let r=0;r<currentMatrix.size;r++){
    const row = document.createElement('tr');
    for(let c=0;c<currentMatrix.size;c++){
      const cell=document.createElement('td');
      cell.textContent=currentMatrix.cells[r][c];
      cell.dataset.row=r;
      cell.dataset.col=c;
      row.appendChild(cell);
    }
    table.appendChild(row);
  }
  setTimeout(()=>table.classList.remove('fade'),50);
  currentPath = [];
  updateStats();
}

document.getElementById('matrix').addEventListener('click', async (e)=>{
  if(e.target.tagName !== 'TD') return;
  const row = parseInt(e.target.dataset.row);
  const col = parseInt(e.target.dataset.col);
  if(row !== currentPath.length) return; // enforce row order
  e.target.classList.add('selected');
  currentPath.push(col);
  const res = await fetch('/api/analyze_path', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:currentPath})});
  const data = await res.json();
  updateStats(data);
});

document.getElementById('random').onclick=async()=>{
  await fetch('/api/random', {method:'POST'});
  await loadMatrix();
};

function updateStats(data){
  const div = document.getElementById('stats');
  if(!data){ div.textContent='Select a path...'; return; }
  div.textContent = `Main diag strength: ${data.main_diagonal_strength.toFixed(2)} | Anti diag strength: ${data.anti_diagonal_strength.toFixed(2)}`;
}

loadMatrix();
</script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get('/api/matrix')
async def get_matrix():
    return JSONResponse(json.loads(matrix.to_json()))

@app.post('/api/random')
async def random_matrix(request: Request):
    global matrix
    matrix = create_random_ekm(default_size)
    return {'status': 'ok'}

@app.post('/api/analyze_path')
async def analyze_path(data: dict):
    path = data.get('path', [])
    analysis = matrix.analyze_path_paradox(path)
    return {
        'main_diagonal_strength': analysis['main_diagonal_strength'],
        'anti_diagonal_strength': analysis['anti_diagonal_strength'],
    }
