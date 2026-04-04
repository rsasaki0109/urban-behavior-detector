// Violation Heatmap using Leaflet.js
(function () {
  'use strict';

  const HEATMAP_DATA_PATH = 'assets/json/heatmap_data.json';

  const VIOLATION_COLORS = {
    walking_smoking: '#ef4444',
    bicycle_phone: '#f97316',
    bicycle_umbrella: '#eab308',
    bicycle_wrong_way: '#e879f9',
    signal_violation: '#f87171',
    sidewalk_riding: '#22d3ee',
  };

  const VIOLATION_LABELS = {
    walking_smoking: 'Walking Smoking',
    bicycle_phone: 'Bicycle Phone',
    bicycle_umbrella: 'Bicycle Umbrella',
    bicycle_wrong_way: 'Wrong Way',
    signal_violation: 'Red Light',
    sidewalk_riding: 'Sidewalk Riding',
  };

  document.addEventListener('DOMContentLoaded', initHeatmap);

  async function initHeatmap() {
    const mapEl = document.getElementById('map');
    if (!mapEl || typeof L === 'undefined') return;

    // Initialize map centered on Tokyo
    const map = L.map('map', {
      scrollWheelZoom: false,
    }).setView([35.68, 139.74], 12);

    // Dark tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    // Load heatmap data
    let data;
    try {
      const resp = await fetch(HEATMAP_DATA_PATH);
      data = await resp.json();
    } catch {
      data = { points: [], total: 0 };
    }

    if (data.points.length === 0) {
      mapEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#888;font-size:0.9rem;">No geo-referenced data available. Set location in config to generate heatmap.</div>';
      return;
    }

    // Group points by location
    const grouped = {};
    for (const p of data.points) {
      const key = p.lat + ',' + p.lon;
      if (!grouped[key]) {
        grouped[key] = { lat: p.lat, lon: p.lon, name: p.location_name, events: [] };
      }
      grouped[key].events.push(p);
    }

    // Add circle markers for each location
    for (const loc of Object.values(grouped)) {
      const counts = {};
      for (const e of loc.events) {
        counts[e.type] = (counts[e.type] || 0) + 1;
      }

      // Size based on event count
      const radius = Math.min(30, 8 + loc.events.length * 3);

      // Color based on most common violation type
      const topType = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      const color = VIOLATION_COLORS[topType] || '#38bdf8';

      const circle = L.circleMarker([loc.lat, loc.lon], {
        radius: radius,
        color: color,
        fillColor: color,
        fillOpacity: 0.35,
        weight: 2,
      }).addTo(map);

      // Popup content
      const popupLines = [
        '<div style="font-family:Inter,sans-serif;font-size:13px;">',
        '<strong>' + (loc.name || 'Camera') + '</strong><br>',
        '<span style="color:#888;">' + loc.events.length + ' violations</span><br><br>',
      ];

      for (const [type, count] of Object.entries(counts).sort((a, b) => b[1] - a[1])) {
        const c = VIOLATION_COLORS[type] || '#999';
        popupLines.push(
          '<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:' + c + ';margin-right:6px;"></span>' +
          (VIOLATION_LABELS[type] || type) + ': <strong>' + count + '</strong><br>'
        );
      }

      popupLines.push('</div>');
      circle.bindPopup(popupLines.join(''));
    }

    // Fit bounds to markers
    const bounds = Object.values(grouped).map(g => [g.lat, g.lon]);
    if (bounds.length > 1) {
      map.fitBounds(bounds, { padding: [40, 40] });
    }
  }
})();
