
/* anyai.js — 小さなふるまい */
(function(){
  // テーマ切替
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const saved = localStorage.getItem('anyai-theme');
  const root = document.documentElement;
  if(saved){
    root.dataset.theme = saved;
  } else if(prefersDark){
    root.dataset.theme = 'dark';
  }
  document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-action="toggle-theme"]');
    if(t){
      const next = (root.dataset.theme === 'dark') ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('anyai-theme', next);
    }
  });

  // タブ（シンプル）
  document.querySelectorAll('[role="tablist"]').forEach(list => {
    list.addEventListener('click', e => {
      const tab = e.target.closest('[role="tab"]');
      if(!tab) return;
      const container = tab.closest('[data-tabs]');
      container.querySelectorAll('[role="tab"]').forEach(t => t.setAttribute('aria-selected', 'false'));
      container.querySelectorAll('[role="tabpanel"]').forEach(p => p.hidden = true);
      tab.setAttribute('aria-selected', 'true');
      const id = tab.getAttribute('aria-controls');
      container.querySelector('#'+id).hidden = false;
    });
  });

  // モバイル・サイドバー開閉（必要時）
  document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-action="toggle-sidebar"]');
    if(!t) return;
    document.body.classList.toggle('sidebar-open');
  });

})();
