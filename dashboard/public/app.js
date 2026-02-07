const skillsBody = document.getElementById("skillsBody");
const refreshBtn = document.getElementById("refreshBtn");
const searchInput = document.getElementById("searchInput");
const statusFilters = Array.from(document.querySelectorAll(".statusFilter"));
const footerNote = document.getElementById("footerNote");

const countTotal = document.getElementById("countTotal");
const countEligible = document.getElementById("countEligible");
const countDisabled = document.getElementById("countDisabled");
const countBlocked = document.getElementById("countBlocked");
const countIneligible = document.getElementById("countIneligible");

const clusterStatus = document.getElementById("clusterStatus");
const clusterMissing = document.getElementById("clusterMissing");
const clusterSource = document.getElementById("clusterSource");

let allSkills = [];
let gatewayStatus = { running: false };
let pendingChanges = new Set(); // 변경된 스킬 추적
let originalStates = new Map(); // 원래 상태 저장

function setStatus(message, tone = "muted") {
  footerNote.textContent = message;
  footerNote.className = `footer ${tone}`;
}

function missingSummary(skill) {
  const missing = skill.missing || {};
  const parts = [];
  if (missing.bins?.length) parts.push(`bins: ${missing.bins.join(", ")}`);
  if (missing.anyBins?.length) parts.push(`anyBins: ${missing.anyBins.join(", ")}`);
  if (missing.env?.length) parts.push(`env: ${missing.env.join(", ")}`);
  if (missing.config?.length) parts.push(`config: ${missing.config.join(", ")}`);
  if (missing.os?.length) parts.push(`os: ${missing.os.join(", ")}`);
  return parts.join("; ");
}

function getMissingHelpLinks(skill) {
  const missing = skill.missing || {};
  const links = [];
  
  // bins (프로그램 설치)
  if (missing.bins?.length) {
    missing.bins.forEach(bin => {
      const help = getBinHelpLink(bin);
      if (help) links.push({ type: 'bin', name: bin, ...help });
    });
  }
  
  // env (환경변수)
  if (missing.env?.length) {
    missing.env.forEach(env => {
      links.push({ 
        type: 'env', 
        name: env, 
        guide: `환경변수 ${env} 설정 필요`,
        action: 'setup'
      });
    });
  }
  
  // config (설정)
  if (missing.config?.length) {
    missing.config.forEach(config => {
      links.push({ 
        type: 'config', 
        name: config, 
        guide: `설정 ${config} 필요`,
        action: 'guide'
      });
    });
  }
  
  return links;
}

function getBinHelpLink(binName) {
  const binMap = {
    'git': { 
      url: 'https://git-scm.com/downloads', 
      guide: 'Git 설치 가이드',
      install: 'winget install Git.Git' 
    },
    'python': { 
      url: 'https://www.python.org/downloads/', 
      guide: 'Python 설치 가이드',
      install: 'winget install Python.Python.3.12' 
    },
    'python3': { 
      url: 'https://www.python.org/downloads/', 
      guide: 'Python 설치 가이드',
      install: 'winget install Python.Python.3.12' 
    },
    'node': { 
      url: 'https://nodejs.org/', 
      guide: 'Node.js 설치 가이드',
      install: 'winget install OpenJS.NodeJS.LTS' 
    },
    'curl': { 
      url: 'https://curl.se/windows/', 
      guide: 'curl 설치 가이드',
      install: 'winget install cURL.cURL' 
    },
    'jq': { 
      url: 'https://stedolan.github.io/jq/download/', 
      guide: 'jq 설치 가이드',
      install: 'winget install stedolan.jq' 
    },
    'uv': { 
      url: 'https://github.com/astral-sh/uv', 
      guide: 'uv 설치 가이드',
      install: 'powershell -c "irm https://astral.sh/uv/install.ps1 | iex"' 
    },
  };
  
  return binMap[binName] || { 
    url: `https://www.google.com/search?q=${encodeURIComponent(binName + ' install windows')}`, 
    guide: `${binName} 설치 가이드`,
    install: null 
  };
}

function statusOf(skill) {
  if (skill.disabled) return "disabled";
  if (skill.blockedByAllowlist) return "blockedByAllowlist";
  if (skill.eligible) return "eligible";
  return "ineligible";
}

function renderClusters(clusters) {
  const statusEntries = Object.entries(clusters.byStatus || {});
  clusterStatus.innerHTML = statusEntries
    .map(([k, list]) => `<div><strong>${k}</strong> (${list.length})</div>`)
    .join("");

  const missingEntries = Object.entries(clusters.byMissingType || {});
  clusterMissing.innerHTML = missingEntries
    .map(([k, list]) => `<div><strong>${k}</strong> (${list.length})</div>`)
    .join("");

  const sourceEntries = Object.entries(clusters.bySource || {});
  clusterSource.innerHTML = sourceEntries
    .map(([k, list]) => `<div><strong>${k}</strong> (${list.length})</div>`)
    .join("");
}

function applyFilters(skills) {
  const query = searchInput.value.trim().toLowerCase();
  const allowedStatuses = new Set(
    statusFilters.filter((f) => f.checked).map((f) => f.value),
  );

  return skills.filter((skill) => {
    const s = statusOf(skill);
    if (!allowedStatuses.has(s)) return false;
    if (!query) return true;
    return (
      skill.name.toLowerCase().includes(query) ||
      skill.description?.toLowerCase().includes(query) ||
      skill.source?.toLowerCase().includes(query)
    );
  });
}

function renderTable(skills) {
  const rows = skills
    .map((skill) => {
      const status = statusOf(skill);
      const missing = missingSummary(skill);
      const enabled = !skill.disabled;
      const disabledFlag = skill.blockedByAllowlist ? "disabled" : "";
      const missingLinks = getMissingHelpLinks(skill);
      const hasMissing = missingLinks.length > 0;
      
      // Missing 컬럼: 요구사항 + 해결 방법 버튼
      let missingCell = missing || "-";
      if (hasMissing) {
        const helpButtons = missingLinks.map(link => {
          if (link.type === 'bin' && link.install) {
            return `<button class="btn-help btn-install" data-bin="${link.name}" data-install="${link.install}" title="${link.guide}">📥 ${link.name} 설치</button>`;
          } else if (link.type === 'bin' && link.url) {
            return `<a href="${link.url}" target="_blank" class="btn-help btn-link" title="${link.guide}">🔗 ${link.name} 다운로드</a>`;
          } else if (link.type === 'env') {
            return `<button class="btn-help btn-env" data-env="${link.name}" title="${link.guide}">⚙️ ${link.name} 설정</button>`;
          } else {
            return `<button class="btn-help btn-guide" data-config="${link.name}" title="${link.guide}">📖 ${link.name} 가이드</button>`;
          }
        }).join(" ");
        missingCell = `<div class="missing-cell">
          <div class="missing-text">${missing}</div>
          <div class="missing-actions">${helpButtons}</div>
        </div>`;
      }
      
      return `
        <tr>
          <td><strong>${skill.emoji || "📦"} ${skill.name}</strong></td>
          <td class="status ${status}" title="${status === 'eligible' ? '필수 요구사항 충족' : status === 'ineligible' ? '필수 요구사항 부족' : status === 'disabled' ? '사용자가 비활성화' : '허용 목록 차단'}">
            ${status === 'eligible' ? '✅ 자격 있음' : status === 'ineligible' ? '❌ 자격 없음' : status === 'disabled' ? '⏸️ 비활성화' : '🚫 차단됨'}
          </td>
          <td>${skill.source || ""}</td>
          <td>${missingCell}</td>
          <td>${skill.description || ""}</td>
          <td>
            <label class="toggle">
              <input type="checkbox" data-skill="${skill.name}" ${enabled ? "checked" : ""} ${disabledFlag} />
              ${enabled ? "On" : "Off"}
            </label>
          </td>
        </tr>
      `;
    })
    .join("");
  skillsBody.innerHTML = rows;
  
  // 버튼 이벤트 리스너 추가
  attachHelpButtonListeners();
}

function updateCounts(skills) {
  const total = skills.length;
  const eligible = skills.filter((s) => s.eligible).length;
  const disabled = skills.filter((s) => s.disabled).length;
  const blocked = skills.filter((s) => s.blockedByAllowlist).length;
  const ineligible = total - eligible - disabled - blocked;

  countTotal.textContent = total;
  countEligible.textContent = eligible;
  countDisabled.textContent = disabled;
  countBlocked.textContent = blocked;
  countIneligible.textContent = ineligible;
}

async function loadData() {
  setStatus("Loading skills...");
  try {
    const [skillsRes, clustersRes] = await Promise.all([
      fetch("/api/skills").catch(() => ({ json: async () => ({ skills: [] }) })),
      fetch("/api/clusters").catch(() => ({ json: async () => ({ byStatus: {}, byMissingType: {}, bySource: {} }) })),
    ]);
    const skillsData = await skillsRes.json();
    const clusters = await clustersRes.json();
    allSkills = skillsData.skills || [];
    updateCounts(allSkills);
    renderClusters(clusters);
    renderTable(applyFilters(allSkills));
    setStatus("Loaded.");
  } catch (err) {
    setStatus(`로드 오류: ${err.message || err}`, "error");
    allSkills = [];
    updateCounts(allSkills);
    renderTable([]);
  }
}

async function checkGatewayStatus() {
  try {
    const res = await fetch("/api/gateway/status");
    gatewayStatus = await res.json();
    updateGatewayUI();
  } catch (err) {
    gatewayStatus = { running: false };
    updateGatewayUI();
  }
}

function updateGatewayUI() {
  const statusEl = document.getElementById("gatewayStatus");
  const restartBtn = document.getElementById("restartGatewayBtn");
  if (statusEl) {
    statusEl.textContent = gatewayStatus.running ? "✅ 실행 중" : "❌ 중지됨";
    statusEl.className = gatewayStatus.running ? "status running" : "status stopped";
  }
  if (restartBtn) {
    restartBtn.disabled = !gatewayStatus.running;
    restartBtn.style.display = gatewayStatus.running ? "inline-block" : "none";
  }
}

async function restartGateway() {
  if (!confirm("게이트웨이를 재시작하시겠습니까? 변경된 스킬 설정이 적용됩니다.")) {
    return;
  }
  setStatus("게이트웨이 재시작 중...", "muted");
  try {
    const res = await fetch("/api/gateway/restart", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json();
    if (res.ok && data.ok) {
      setStatus("게이트웨이 재시작 완료!", "success");
      setTimeout(() => checkGatewayStatus(), 2000);
    } else {
      setStatus(data.error || "게이트웨이 재시작 실패", "error");
    }
  } catch (err) {
    setStatus(`오류: ${err.message || err}`, "error");
  }
}

async function toggleSkill(name, enabled) {
  // 원래 상태 저장 (처음 변경 시)
  if (!originalStates.has(name)) {
    const skill = allSkills.find(s => s.name === name);
    if (skill) {
      originalStates.set(name, !skill.disabled);
    }
  }
  
  // 변경 추적
  const original = originalStates.get(name);
  if (enabled === original) {
    pendingChanges.delete(name);
  } else {
    pendingChanges.add(name);
  }
  
  updateActionButtons();
  setStatus(`${name} ${enabled ? "활성화" : "비활성화"} 예정 (적용 버튼을 눌러 저장하세요)`, "muted");
}

async function applyChanges() {
  if (pendingChanges.size === 0) {
    setStatus("변경사항이 없습니다.", "muted");
    return;
  }
  
  if (!confirm(`${pendingChanges.size}개의 스킬 설정을 적용하시겠습니까?`)) {
    return;
  }
  
  setStatus("설정 적용 중...", "muted");
  const changes = Array.from(pendingChanges);
  
  try {
    for (const name of changes) {
      const skill = allSkills.find(s => s.name === name);
      if (!skill) continue;
      
      const enabled = !skill.disabled;
      const res = await fetch("/api/skills/toggle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, enabled }),
      });
      
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `Failed to update ${name}`);
      }
    }
    
    // 변경사항 초기화
    pendingChanges.clear();
    originalStates.clear();
    updateActionButtons();
    
    // 데이터 다시 로드
    await loadData();
    
    if (gatewayStatus.running) {
      setStatus(`✅ ${changes.length}개 스킬 설정 적용됨. 게이트웨이 재시작 버튼을 눌러 적용하세요.`, "success");
    } else {
      setStatus(`✅ ${changes.length}개 스킬 설정 적용됨. 게이트웨이 시작 시 적용됩니다.`, "success");
    }
  } catch (err) {
    setStatus(`오류: ${err.message || err}`, "error");
  }
}

function cancelChanges() {
  if (pendingChanges.size === 0) {
    setStatus("변경사항이 없습니다.", "muted");
    return;
  }
  
  if (!confirm("변경사항을 취소하시겠습니까?")) {
    return;
  }
  
  // 원래 상태로 복원
  pendingChanges.forEach(name => {
    const original = originalStates.get(name);
    const checkbox = document.querySelector(`input[data-skill="${name}"]`);
    if (checkbox) {
      checkbox.checked = original;
    }
  });
  
  pendingChanges.clear();
  originalStates.clear();
  updateActionButtons();
  setStatus("변경사항이 취소되었습니다.", "muted");
  
  // 데이터 다시 로드하여 원래 상태 확인
  loadData();
}

function updateActionButtons() {
  const applyBtn = document.getElementById("applyBtn");
  const cancelBtn = document.getElementById("cancelBtn");
  
  if (pendingChanges.size > 0) {
    if (applyBtn) {
      applyBtn.style.display = "inline-block";
      applyBtn.textContent = `적용 (${pendingChanges.size})`;
    }
    if (cancelBtn) {
      cancelBtn.style.display = "inline-block";
    }
  } else {
    if (applyBtn) applyBtn.style.display = "none";
    if (cancelBtn) cancelBtn.style.display = "none";
  }
}

skillsBody.addEventListener("change", (event) => {
  const target = event.target;
  if (target?.matches("input[type=checkbox][data-skill]")) {
    const name = target.getAttribute("data-skill");
    toggleSkill(name, target.checked);
  }
});

refreshBtn.addEventListener("click", loadData);
searchInput.addEventListener("input", () => renderTable(applyFilters(allSkills)));
statusFilters.forEach((f) =>
  f.addEventListener("change", () => renderTable(applyFilters(allSkills))),
);

const restartBtn = document.getElementById("restartGatewayBtn");
if (restartBtn) {
  restartBtn.addEventListener("click", restartGateway);
}

const applyBtn = document.getElementById("applyBtn");
if (applyBtn) {
  applyBtn.addEventListener("click", applyChanges);
}

const cancelBtn = document.getElementById("cancelBtn");
if (cancelBtn) {
  cancelBtn.addEventListener("click", cancelChanges);
}

// 탭 전환
const tabInstalled = document.getElementById("tabInstalled");
const tabClawHub = document.getElementById("tabClawHub");
const installedControls = document.getElementById("installedControls");
const clawhubControls = document.getElementById("clawhubControls");
const installedTable = document.getElementById("installedTable");
const clawhubTable = document.getElementById("clawhubTable");

let clawhubResults = [];
let selectedClawhubSkill = null;
let clawhubCurrentPage = 1;
let clawhubTotalPages = 1;
let clawhubTotal = 0;
let clawhubLimit = 20;

if (tabInstalled && tabClawHub) {
  tabInstalled.addEventListener("click", () => {
    tabInstalled.classList.add("active");
    tabClawHub.classList.remove("active");
    installedControls.style.display = "flex";
    clawhubControls.style.display = "none";
    installedTable.style.display = "block";
    clawhubTable.style.display = "none";
  });

  tabClawHub.addEventListener("click", () => {
    tabClawHub.classList.add("active");
    tabInstalled.classList.remove("active");
    installedControls.style.display = "none";
    clawhubControls.style.display = "flex";
    installedTable.style.display = "none";
    clawhubTable.style.display = "block";
    // ClawHub 탭이 열릴 때 자동으로 목록 로드 (검색어 없이)
    if (clawhubResults.length === 0) {
      loadClawHubList();
    }
  });
}

// ClawHub 검색
const clawhubSearchInput = document.getElementById("clawhubSearchInput");
const clawhubSearchBtn = document.getElementById("clawhubSearchBtn");
const clawhubInstallBtn = document.getElementById("clawhubInstallBtn");
const clawhubBody = document.getElementById("clawhubBody");
const clawhubInstallNotice = document.getElementById("clawhubInstallNotice");
const installClawhubBtn = document.getElementById("installClawhubBtn");
const clawhubInfo = document.getElementById("clawhubInfo");

async function loadClawHubList(page = 1) {
  const query = clawhubSearchInput ? clawhubSearchInput.value.trim() : "";
  clawhubCurrentPage = page;
  
  setStatus(query ? "ClawHub 검색 중..." : "ClawHub 스킬 목록 로드 중...", "muted");
  try {
    const url = `/api/clawhub/search?page=${page}&limit=${clawhubLimit}${query ? `&q=${encodeURIComponent(query)}` : ""}`;
    const res = await fetch(url);
    
    // Response body는 한 번만 읽을 수 있으므로, 먼저 텍스트로 읽고 JSON 파싱 시도
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch (jsonErr) {
      // "Not found" 같은 단순 텍스트 응답도 처리
      if (text.trim() === "Not found" || text.trim().toLowerCase() === "not found") {
        data = {
          results: [],
          total: 0,
          page: 1,
          limit: clawhubLimit,
          totalPages: 0,
          error: "No results found"
        };
      } else {
        throw new Error(`서버 응답 파싱 오류: ${text.substring(0, 200)}`);
      }
    }

    if (data?.error === "Not found") {
      clawhubBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--muted);">
        ClawHub API가 없습니다. ghostchat/dashboard 서버로 실행했는지 확인하세요.
      </td></tr>`;
      setStatus("ClawHub API 없음: 올바른 대시보드 서버로 실행하세요.", "error");
      return;
    }

    if (res.status === 503 && data.needsInstall) {
      let diagHtml = "";
      if (data.diagnostics) {
        const diag = data.diagnostics;
        const resolved = diag.resolved ? `${diag.resolved.cmd} ${Array.isArray(diag.resolved.args) ? diag.resolved.args.join(" ") : ""}`.trim() : "없음";
        const which = (diag.which || "").replace(/\r?\n/g, "<br/>");
        const candidates = (diag.candidates || []).join("<br/>");
        diagHtml = `
          <div style="margin-top: 12px; font-size: 11px; color: var(--muted); text-align: left;">
            <div><strong>진단</strong></div>
            <div>resolved: ${resolved}</div>
            <div>where clawhub: ${which || "-"}</div>
            <div>candidates:<br/>${candidates || "-"}</div>
          </div>
        `;
      }
      clawhubInstallNotice.style.display = "block";
      clawhubBody.innerHTML = `<tr><td colspan="5" style="text-align: center; padding: 40px; color: var(--muted);">
        <div style="margin-bottom: 16px;">⚠️ ClawHub CLI가 설치되지 않았습니다</div>
        <div style="font-size: 12px; margin-bottom: 12px;">ClawHub 스킬을 검색하고 설치하려면 ClawHub CLI가 필요합니다.</div>
        <div style="font-size: 11px; color: var(--accent);">설치 명령어: <code>npm i -g clawhub</code></div>
        ${diagHtml}
      </td></tr>`;
      setStatus("ClawHub CLI가 설치되지 않았습니다. 설치 후 다시 시도하세요.", "error");
      return;
    }

    if (!res.ok && data?.error && data.error !== "No results found") {
      throw new Error(data.error || "로드 실패");
    }

    clawhubInstallNotice.style.display = "none";
    clawhubResults = data.results || [];
    clawhubTotal = data.total || 0;
    clawhubTotalPages = data.totalPages || 1;
    clawhubCurrentPage = data.page || 1;
    
    renderClawHubResults(clawhubResults);
    renderClawHubPagination();
    
    if (clawhubTotal === 0) {
      setStatus("결과가 없습니다.", "muted");
    } else {
      const statusMsg = query 
        ? `${clawhubTotal}개 스킬 발견 (페이지 ${clawhubCurrentPage}/${clawhubTotalPages})`
        : `${clawhubTotal}개 스킬 (페이지 ${clawhubCurrentPage}/${clawhubTotalPages})`;
      setStatus(statusMsg, "success");
    }
  } catch (err) {
    setStatus(`로드 오류: ${err.message || err}`, "error");
    clawhubBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--muted);">로드 실패: ${err.message || err}</td></tr>`;
  }
}

async function loadDashboardInfo() {
  if (!clawhubInfo) return;
  try {
    const res = await fetch("/api/info");
    if (!res.ok) return;
    const data = await res.json();
    if (data?.name) {
      clawhubInfo.textContent = `Server: ${data.name} (${data.version})`;
    }
  } catch {
    // ignore
  }
}

async function searchClawHub() {
  clawhubCurrentPage = 1; // 검색 시 첫 페이지로
  await loadClawHubList(1);
}

function renderClawHubResults(results) {
  if (results.length === 0) {
    clawhubBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--muted);">결과가 없습니다.</td></tr>`;
    clawhubInstallBtn.style.display = "none";
    return;
  }

  const rows = results.map((skill) => {
    const slug = skill.slug || skill.name || "";
    const version = skill.version || skill.latestVersion || "-";
    const description = skill.description || skill.summary || "";
    const tags = (skill.tags || []).slice(0, 5).join(", ") || "-";
    const isInstalled = allSkills.some(s => s.name === slug);

    return `
      <tr>
        <td><strong>${skill.emoji || "📦"} ${skill.name || slug}</strong></td>
        <td>${version}</td>
        <td>${description}</td>
        <td>${tags}</td>
        <td>
          ${isInstalled 
            ? '<span style="color: var(--muted);">이미 설치됨</span>' 
            : `<button class="btn-install-skill" data-slug="${slug}">📥 설치</button>`
          }
        </td>
      </tr>
    `;
  }).join("");

  clawhubBody.innerHTML = rows;

  // 설치 버튼 이벤트 리스너
  document.querySelectorAll('.btn-install-skill').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const slug = e.target.getAttribute('data-slug');
      await installClawHubSkill(slug);
    });
  });
}

function renderClawHubPagination() {
  let paginationEl = document.getElementById("clawhubPagination");
  if (!paginationEl) {
    // 페이징 컨테이너가 없으면 생성
    paginationEl = document.createElement("div");
    paginationEl.id = "clawhubPagination";
    paginationEl.className = "pagination";
    clawhubTable.appendChild(paginationEl);
  }

  if (clawhubTotalPages <= 1) {
    paginationEl.style.display = "none";
    return;
  }

  paginationEl.style.display = "flex";
  paginationEl.style.gap = "8px";
  paginationEl.style.alignItems = "center";
  paginationEl.style.justifyContent = "center";
  paginationEl.style.padding = "20px";
  paginationEl.style.background = "var(--panel-2)";
  paginationEl.style.borderRadius = "10px";
  paginationEl.style.marginTop = "16px";

  const prevDisabled = clawhubCurrentPage <= 1;
  const nextDisabled = clawhubCurrentPage >= clawhubTotalPages;

  // 페이지 번호 범위 계산 (현재 페이지 주변 5개)
  const startPage = Math.max(1, clawhubCurrentPage - 2);
  const endPage = Math.min(clawhubTotalPages, clawhubCurrentPage + 2);
  const pageNumbers = [];
  for (let i = startPage; i <= endPage; i++) {
    pageNumbers.push(i);
  }

  paginationEl.innerHTML = `
    <button class="btn-pagination" ${prevDisabled ? 'disabled' : ''} data-page="${clawhubCurrentPage - 1}">◀ 이전</button>
    ${startPage > 1 ? `<button class="btn-pagination" data-page="1">1</button>${startPage > 2 ? '<span style="color: var(--muted);">...</span>' : ''}` : ''}
    ${pageNumbers.map(page => `
      <button class="btn-pagination ${page === clawhubCurrentPage ? 'active' : ''}" data-page="${page}">${page}</button>
    `).join('')}
    ${endPage < clawhubTotalPages ? `${endPage < clawhubTotalPages - 1 ? '<span style="color: var(--muted);">...</span>' : ''}<button class="btn-pagination" data-page="${clawhubTotalPages}">${clawhubTotalPages}</button>` : ''}
    <button class="btn-pagination" ${nextDisabled ? 'disabled' : ''} data-page="${clawhubCurrentPage + 1}">다음 ▶</button>
    <span style="margin-left: 16px; color: var(--muted); font-size: 13px;">총 ${clawhubTotal}개</span>
  `;

  // 페이지 버튼 이벤트 리스너
  paginationEl.querySelectorAll('.btn-pagination').forEach(btn => {
    if (!btn.disabled) {
      btn.addEventListener('click', (e) => {
        const page = parseInt(e.target.getAttribute('data-page'));
        if (page && page !== clawhubCurrentPage) {
          loadClawHubList(page);
        }
      });
    }
  });
}

async function installClawHubSkill(slug) {
  if (!confirm(`"${slug}" 스킬을 설치하시겠습니까?`)) {
    return;
  }

  setStatus(`"${slug}" 설치 중...`, "muted");
  try {
    const res = await fetch("/api/clawhub/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slug }),
    });

    const data = await res.json();

    if (res.status === 503 && data.needsInstall) {
      setStatus("ClawHub CLI가 설치되지 않았습니다. 먼저 ClawHub CLI를 설치하세요.", "error");
      clawhubInstallNotice.style.display = "block";
      return;
    }

    if (!res.ok) {
      throw new Error(data.error || "설치 실패");
    }

    setStatus(`✅ "${slug}" 설치 완료! 새로고침하여 확인하세요.`, "success");
    
    // 설치된 스킬 목록 새로고침
    setTimeout(() => {
      loadData();
      // 설치된 스킬 탭으로 전환
      if (tabInstalled) tabInstalled.click();
    }, 2000);
  } catch (err) {
    setStatus(`설치 오류: ${err.message || err}`, "error");
  }
}

if (clawhubSearchBtn) {
  clawhubSearchBtn.addEventListener("click", searchClawHub);
}

if (clawhubSearchInput) {
  clawhubSearchInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      searchClawHub();
    }
  });
}

if (installClawhubBtn) {
  installClawhubBtn.addEventListener("click", async () => {
    const installCmd = "npm i -g clawhub";
    if (confirm(`ClawHub CLI를 설치하시겠습니까?\n\n명령어: ${installCmd}\n\n새 터미널 창에서 실행됩니다.`)) {
      setStatus("ClawHub CLI 설치 중...", "muted");
      try {
        const result = await fetch('/api/install-bin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ bin: 'clawhub', installCmd }),
        });
        const data = await result.json();
        if (result.ok) {
          setStatus(`✅ ClawHub CLI 설치 명령 실행됨. 설치 완료 후 새로고침하세요.`, "success");
        } else {
          setStatus(`❌ 설치 실패: ${data.error || 'Unknown error'}`, "error");
        }
      } catch (err) {
        setStatus(`📋 설치 명령어를 클립보드에 복사했습니다. 새 터미널에서 실행하세요: ${installCmd}`, "muted");
        if (navigator.clipboard) {
          navigator.clipboard.writeText(installCmd);
        }
      }
    }
  });
}

// 초기 로드 (게이트웨이 상태 확인은 선택적)
Promise.all([
  loadData(),
  checkGatewayStatus().catch(() => {
    // 게이트웨이 상태 확인 실패해도 계속 진행 (게이트웨이 없어도 스킬 목록은 볼 수 있음)
    gatewayStatus = { running: false };
    updateGatewayUI();
  }),
]).catch((err) => {
  setStatus(`Error: ${err.message || err}`, "muted");
});

loadDashboardInfo();

// 주기적으로 게이트웨이 상태 확인 (30초마다) - 실패해도 무시
setInterval(() => {
  checkGatewayStatus().catch(() => {
    // 게이트웨이 상태 확인 실패는 무시 (게이트웨이 없어도 스킬 목록은 볼 수 있음)
  });
}, 30000);

function attachHelpButtonListeners() {
  // 설치 버튼 (winget 등)
  document.querySelectorAll('.btn-install').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const bin = e.target.getAttribute('data-bin');
      const installCmd = e.target.getAttribute('data-install');
      if (confirm(`${bin}을(를) 설치하시겠습니까?\n\n명령어: ${installCmd}\n\n새 터미널 창에서 실행됩니다.`)) {
        setStatus(`${bin} 설치 중...`, "muted");
        try {
          // 새 터미널 창에서 설치 명령 실행
          const result = await fetch('/api/install-bin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bin, installCmd }),
          });
          const data = await result.json();
          if (result.ok) {
            setStatus(`✅ ${bin} 설치 명령 실행됨. 설치 완료 후 새로고침하세요.`, "success");
            setTimeout(() => loadData(), 5000);
          } else {
            setStatus(`❌ 설치 실패: ${data.error || 'Unknown error'}`, "error");
          }
        } catch (err) {
          // 클라이언트에서 직접 실행 (fallback)
          setStatus(`📋 설치 명령어를 클립보드에 복사했습니다. 새 터미널에서 실행하세요: ${installCmd}`, "muted");
          if (navigator.clipboard) {
            navigator.clipboard.writeText(installCmd);
          }
        }
      }
    });
  });
  
  // 환경변수 설정 버튼
  document.querySelectorAll('.btn-env').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const env = e.target.getAttribute('data-env');
      const guide = showEnvSetupGuide(env);
      if (guide) {
        setStatus(`📖 ${env} 설정 가이드: ${guide}`, "muted");
      }
    });
  });
  
  // 가이드 버튼
  document.querySelectorAll('.btn-guide').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const config = e.target.getAttribute('data-config');
      setStatus(`📖 설정 가이드: ${config}`, "muted");
      // TODO: 설정 가이드 모달 표시
    });
  });
}

function showEnvSetupGuide(envName) {
  const guides = {
    'OPENAI_API_KEY': 'OpenAI API 키 발급: https://platform.openai.com/api-keys',
    'GOOGLE_PLACES_API_KEY': 'Google Places API 키 발급: https://console.cloud.google.com/apis/credentials',
    'GEMINI_API_KEY': 'Gemini API 키 발급: https://aistudio.google.com/app/apikey',
    'ELEVENLABS_API_KEY': 'ElevenLabs API 키 발급: https://elevenlabs.io/app/settings/api-keys',
    'NOTION_API_KEY': 'Notion API 키 발급: https://www.notion.so/my-integrations',
    'TRELLO_API_KEY': 'Trello API 키 발급: https://trello.com/app-key',
  };
  
  const guide = guides[envName] || `환경변수 ${envName} 설정 필요`;
  
  // 모달 또는 안내 표시
  const modal = document.createElement('div');
  modal.className = 'help-modal';
  modal.innerHTML = `
    <div class="help-modal-content">
      <h3>${envName} 설정 가이드</h3>
      <p>${guide}</p>
      <div class="help-commands">
        <p><strong>PowerShell에서 설정:</strong></p>
        <code>[System.Environment]::SetEnvironmentVariable("${envName}", "your-value", "User")</code>
        <p><strong>또는 임시 설정:</strong></p>
        <code>$env:${envName} = "your-value"</code>
      </div>
      <button class="btn-close-modal">닫기</button>
    </div>
  `;
  document.body.appendChild(modal);
  
  modal.querySelector('.btn-close-modal').addEventListener('click', () => {
    modal.remove();
  });
  
  return guide;
}
