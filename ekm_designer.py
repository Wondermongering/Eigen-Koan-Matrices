from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import json

from eigen_koan_matrix import create_random_ekm, EigenKoanMatrix

app = FastAPI(title="EKM Designer")

# In-memory matrix state
default_size = 4
matrix: EigenKoanMatrix = create_random_ekm(default_size)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>EKM Designer</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
#matrix td { border: 1px solid #666; padding: 6px; min-width: 80px; text-align: center; cursor: move; }
#matrix th { border: 1px solid #666; padding: 6px; background:#f0f0f0; }
.drag-over { background-color: #ffe0b2; }
.selected { background-color: #b2dfdb; }
</style>
</head>
<body>
<h1>Interactive EKM Designer</h1>
<table id='matrix'></table>
<div id='analysis'></div>
<script>
let currentMatrix=null;
let dragSrc=null;
async function loadMatrix(){
  const res = await fetch('/api/designer_matrix');
  currentMatrix = await res.json();
  const table = document.getElementById('matrix');
  table.innerHTML='';
  const header = document.createElement('tr');
  header.appendChild(document.createElement('th'));
  for(let c=0;c<currentMatrix.size;c++){
    const th=document.createElement('th');
    th.textContent=currentMatrix.constraint_cols[c];
    th.dataset.col=c;
    th.draggable=true;
    th.addEventListener('dragstart',startDragCol);
    th.addEventListener('dragover',dragOver);
    th.addEventListener('drop',dropCol);
    header.appendChild(th);
  }
  table.appendChild(header);
  for(let r=0;r<currentMatrix.size;r++){
    const row=document.createElement('tr');
    const th=document.createElement('th');
    th.textContent=currentMatrix.task_rows[r];
    th.dataset.row=r;
    th.draggable=true;
    th.addEventListener('dragstart',startDragRow);
    th.addEventListener('dragover',dragOver);
    th.addEventListener('drop',dropRow);
    row.appendChild(th);
    for(let c=0;c<currentMatrix.size;c++){
      const cell=document.createElement('td');
      cell.textContent=currentMatrix.cells[r][c];
      cell.dataset.row=r;
      cell.dataset.col=c;
      cell.draggable=true;
      cell.addEventListener('dragstart',startDragCell);
      cell.addEventListener('dragover',dragOver);
      cell.addEventListener('drop',dropCell);
      row.appendChild(cell);
    }
    table.appendChild(row);
  }
}
function dragOver(e){
  e.preventDefault();
  e.currentTarget.classList.add('drag-over');
}
function clearDrag(){
  document.querySelectorAll('.drag-over').forEach(el=>el.classList.remove('drag-over'));
}
function startDragCell(e){
  dragSrc={type:'cell',row:e.target.dataset.row,col:e.target.dataset.col};
}
async function dropCell(e){
  e.preventDefault();
  clearDrag();
  if(!dragSrc || dragSrc.type!=='cell') return;
  const destRow=e.target.dataset.row;
  const destCol=e.target.dataset.col;
  await fetch('/api/update_cell',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({src:{row:dragSrc.row,col:dragSrc.col},dest:{row:destRow,col:destCol}})});
  await loadMatrix();
  dragSrc=null;
}
function startDragRow(e){
  dragSrc={type:'row',row:e.target.dataset.row};
}
async function dropRow(e){
  e.preventDefault();
  clearDrag();
  if(!dragSrc||dragSrc.type!=='row') return;
  const destRow=e.target.dataset.row;
  await fetch('/api/swap_tasks',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({row1:dragSrc.row,row2:destRow})});
  await loadMatrix();
  dragSrc=null;
}
function startDragCol(e){
  dragSrc={type:'col',col:e.target.dataset.col};
}
async function dropCol(e){
  e.preventDefault();
  clearDrag();
  if(!dragSrc||dragSrc.type!=='col') return;
  const destCol=e.target.dataset.col;
  await fetch('/api/swap_constraints',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({col1:dragSrc.col,col2:destCol})});
  await loadMatrix();
  dragSrc=null;
}
async function analyzePath(path){
  const res=await fetch('/api/analyze_path',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});
  return await res.json();
}
let currentPath=[];
document.getElementById('matrix').addEventListener('click',async(e)=>{
  if(e.target.tagName!=='TD') return;
  const row=parseInt(e.target.dataset.row);
  const col=parseInt(e.target.dataset.col);
  if(row!==currentPath.length){currentPath=[];document.querySelectorAll('td').forEach(td=>td.classList.remove('selected'));}
  e.target.classList.add('selected');
  currentPath.push(col);
  const data=await analyzePath(currentPath);
  document.getElementById('analysis').textContent=`Tensions: ${data.tension_count}, Main diag: ${data.main_diagonal_strength.toFixed(2)}`;
});
loadMatrix();
</script>
</body>
</html>
"""

@app.get('/', response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get('/api/designer_matrix')
async def get_matrix():
    return JSONResponse(json.loads(matrix.to_json()))

@app.post('/api/update_cell')
async def update_cell(data: dict):
    src=data['src']; dest=data['dest']
    content=matrix.get_cell(int(src['row']),int(src['col']))
    matrix.set_cell(int(src['row']),int(src['col']), matrix.get_cell(int(dest['row']),int(dest['col'])))
    matrix.set_cell(int(dest['row']),int(dest['col']), content)
    return {'status':'ok'}

@app.post('/api/swap_tasks')
async def swap_tasks(data: dict):
    r1=int(data['row1']); r2=int(data['row2'])
    matrix.task_rows[r1], matrix.task_rows[r2] = matrix.task_rows[r2], matrix.task_rows[r1]
    matrix.cells[r1], matrix.cells[r2] = matrix.cells[r2], matrix.cells[r1]
    return {'status':'ok'}

@app.post('/api/swap_constraints')
async def swap_constraints(data: dict):
    c1=int(data['col1']); c2=int(data['col2'])
    matrix.constraint_cols[c1], matrix.constraint_cols[c2] = matrix.constraint_cols[c2], matrix.constraint_cols[c1]
    for row in matrix.cells:
        row[c1], row[c2] = row[c2], row[c1]
    return {'status':'ok'}

@app.post('/api/analyze_path')
async def analyze_path(data: dict):
    path=data.get('path',[])
    analysis=matrix.analyze_path_paradox(path)
    difficulty=analysis['tension_count'] + (1-analysis['main_diagonal_strength']) + (1-analysis['anti_diagonal_strength'])
    analysis['predicted_difficulty']=difficulty
    return analysis
