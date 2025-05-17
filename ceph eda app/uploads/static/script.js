// static/script.js
function toggleView(viewId) {
    const views = ['eda', 'recovery', 'replication'];
    views.forEach(id => {
        document.getElementById(id).style.display = (id === viewId) ? 'block' : 'none';
    });
}
