// Urban Behavior Detector - Demo Site
(function () {
  'use strict';

  const DEMO_DATA_PATH = 'assets/json/demo_index.json';

  const VIOLATION_LABELS = {
    walking_smoking: 'Walking Smoking',
    bicycle_phone: 'Bicycle Phone',
    bicycle_umbrella: 'Bicycle Umbrella',
    bicycle_wrong_way: 'Wrong Way',
    walking_phone: 'Walking Phone',
    signal_violation: 'Red Light',
    sidewalk_riding: 'Sidewalk Riding',
  };

  let demoData = [];
  let activeVideoIdx = 0;
  let activeFilter = 'all';

  // -- Init --
  document.addEventListener('DOMContentLoaded', () => {
    initNav();
    initFilters();
    loadDemoData();
  });

  // -- Navigation --
  function initNav() {
    const toggle = document.querySelector('.nav-toggle');
    const links = document.querySelector('.nav-links');
    if (toggle && links) {
      toggle.addEventListener('click', () => links.classList.toggle('open'));
      links.querySelectorAll('a').forEach(a =>
        a.addEventListener('click', () => links.classList.remove('open'))
      );
    }
  }

  // -- Load demo data --
  async function loadDemoData() {
    try {
      const resp = await fetch(DEMO_DATA_PATH);
      demoData = await resp.json();
    } catch {
      // Fallback: use embedded sample data
      demoData = getEmbeddedData();
    }
    renderTabs();
    selectVideo(0);
  }

  // -- Tabs --
  function renderTabs() {
    const container = document.getElementById('demo-tabs');
    if (!container) return;
    container.innerHTML = '';
    demoData.forEach((d, i) => {
      const btn = document.createElement('button');
      btn.className = 'demo-tab' + (i === 0 ? ' active' : '');
      btn.textContent = d.video_id.replace(/_/g, ' ');
      btn.addEventListener('click', () => selectVideo(i));
      container.appendChild(btn);
    });
  }

  // -- Filters --
  function initFilters() {
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeFilter = btn.dataset.filter;
        selectVideo(activeVideoIdx);
      });
    });
  }

  function getFilteredEvents(data) {
    if (activeFilter === 'all') return data.events;
    return data.events.filter(e => e.type === activeFilter);
  }

  function selectVideo(idx) {
    activeVideoIdx = idx;
    const data = demoData[idx];

    // Update tab states
    document.querySelectorAll('.demo-tab').forEach((tab, i) => {
      tab.classList.toggle('active', i === idx);
    });

    renderEventLog(data);
    renderTimeline(data);
    renderJsonPreview(data);
    renderStats(data);
    updateVideoPlayer(data);
  }

  // -- Video player --
  function updateVideoPlayer(data) {
    const video = document.getElementById('demo-video');
    const placeholder = document.getElementById('video-placeholder');
    if (!video || !placeholder) return;

    // Try to load the video file
    const videoPath = 'assets/videos/' + data.video_id + '_detected.mp4';

    // Test if video exists by trying to load it
    video.onloadeddata = function () {
      video.style.display = 'block';
      placeholder.style.display = 'none';
    };

    video.onerror = function () {
      video.style.display = 'none';
      placeholder.style.display = '';
      placeholder.innerHTML = `
        <div style="position:relative;z-index:1;text-align:center;padding:20px;">
          <div style="font-size:2.5rem;margin-bottom:8px;">&#9654;</div>
          <div style="font-size:0.9rem;color:var(--text-secondary);">${data.video_id.replace(/_/g, ' ')}</div>
          <div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;">${data.resolution} | ${data.total_frames} frames | ${data.fps} fps</div>
          <div style="font-size:0.75rem;color:var(--text-muted);margin-top:8px;">Run the pipeline to generate annotated video</div>
        </div>
      `;
    };

    video.src = videoPath;
  }

  function seekVideo(time) {
    const video = document.getElementById('demo-video');
    if (video && video.style.display !== 'none') {
      video.currentTime = time;
      video.play().catch(() => {});
    }
  }

  // -- Event log --
  function renderEventLog(data) {
    const list = document.getElementById('event-list');
    const count = document.getElementById('event-count');
    if (!list || !count) return;

    const events = getFilteredEvents(data);
    count.textContent = events.length + ' events';

    if (events.length === 0) {
      list.innerHTML = '<div style="padding:24px;text-align:center;color:var(--text-muted);font-size:0.85rem;">No violations detected in this footage (VLM-verified).<br>Person/vehicle detection &amp; tracking are active &mdash; see the annotated video above.<br><br><span style="font-size:0.7rem;">Note: 480p live camera resolution is insufficient for reliable cigarette/phone detection.<br>Higher resolution input (&ge;720p, closer camera angle) is recommended for violation detection.</span></div>';
      return;
    }

    list.innerHTML = events.map(e => `
      <div class="event-item" data-start-time="${e.start_time}" data-snapshot="${e.snapshot || ''}" title="Click to seek to ${formatTime(e.start_time)}">
        <div class="event-row">
          <span class="event-badge ${e.type}">${VIOLATION_LABELS[e.type] || e.type}</span>
          <span class="event-info">
            <strong>Track #${e.track_id}</strong> &middot;
            ${formatTime(e.start_time)} &ndash; ${formatTime(e.end_time)}
          </span>
          <span class="event-conf">${(e.confidence * 100).toFixed(0)}%</span>
        </div>
        ${e.snapshot ? `<div class="event-snapshot-wrapper"><img class="event-snapshot" src="${e.snapshot}" alt="Detection frame" loading="lazy"></div>` : ''}
        ${e.vlm_evaluation ? `<div class="vlm-eval">
          <span class="vlm-icon">${e.vlm_evaluation.smoking_detected ? '&#9989;' : '&#10060;'}</span>
          <span class="vlm-label">VLM:</span>
          <span class="vlm-desc">${e.vlm_evaluation.description || ''}</span>
        </div>` : ''}
      </div>
    `).join('');

    // Add click-to-seek handlers
    list.querySelectorAll('.event-item').forEach(item => {
      item.addEventListener('click', () => {
        const time = parseFloat(item.dataset.startTime);
        if (!isNaN(time)) seekVideo(time);
      });
    });
  }

  // -- Timeline --
  function renderTimeline(data) {
    const bar = document.getElementById('timeline-bar');
    const labels = document.getElementById('timeline-labels');
    if (!bar) return;

    const events = getFilteredEvents(data);
    const totalFrames = data.total_frames;
    const duration = totalFrames / data.fps;

    bar.innerHTML = events.map(e => {
      const left = (e.start_frame / totalFrames) * 100;
      const width = Math.max(((e.end_frame - e.start_frame) / totalFrames) * 100, 0.8);
      return `<div class="timeline-event ${e.type}"
        style="left:${left}%;width:${width}%;cursor:pointer;"
        data-start-time="${e.start_time}"
        title="${VIOLATION_LABELS[e.type] || e.type}: Track #${e.track_id} (${formatTime(e.start_time)}-${formatTime(e.end_time)})"></div>`;
    }).join('');

    // Timeline event click-to-seek
    bar.querySelectorAll('.timeline-event').forEach(el => {
      el.addEventListener('click', () => {
        const time = parseFloat(el.dataset.startTime);
        if (!isNaN(time)) seekVideo(time);
      });
    });

    if (labels) {
      labels.innerHTML = `<span>0:00</span><span>${formatTime(duration)}</span>`;
    }
  }

  // -- JSON Preview --
  function renderJsonPreview(data) {
    const el = document.getElementById('json-body');
    if (!el) return;

    const sample = {
      video_id: data.video_id,
      events: data.events.slice(0, 2),
    };

    el.innerHTML = '<pre>' + syntaxHighlight(JSON.stringify(sample, null, 2)) + '</pre>';
  }

  function syntaxHighlight(json) {
    return json.replace(/("(\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
      function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
          if (/:$/.test(match)) {
            cls = 'json-key';
            match = match.replace(/:$/, '') + ':';
          } else {
            cls = 'json-string';
          }
        }
        return '<span class="' + cls + '">' + match + '</span>';
      }
    );
  }

  // -- Stats --
  function renderStats(data) {
    const body = document.getElementById('stats-body');
    if (!body) return;

    const counts = {};
    for (const e of data.events) {
      counts[e.type] = (counts[e.type] || 0) + 1;
    }

    const types = ['walking_smoking', 'walking_phone', 'bicycle_phone', 'bicycle_umbrella',
                   'bicycle_wrong_way', 'signal_violation', 'sidewalk_riding'];
    if (data.events.length === 0) {
      body.innerHTML = '<div class="stat-card"><div class="stat-value" style="color:var(--green);">0</div><div class="stat-label">No violations detected</div></div>';
      return;
    }

    body.innerHTML = [
      { label: 'Total Events', value: data.events.length, cls: '' },
      ...types
        .filter(t => counts[t])
        .map(t => ({
          label: VIOLATION_LABELS[t] || t,
          value: counts[t] || 0,
          cls: t,
        }))
    ].map(s => `
      <div class="stat-card">
        <div class="stat-value ${s.cls}">${s.value}</div>
        <div class="stat-label">${s.label}</div>
      </div>
    `).join('');
  }

  // -- Helpers --
  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + String(s).padStart(2, '0');
  }

  function getEmbeddedData() {
    return [
      {
        video_id: "kabukicho_3m",
        video_file: "kabukicho_3m.mp4",
        fps: 30.0,
        total_frames: 5400,
        resolution: "854x480",
        events: []
      }
    ];
  }
})();
