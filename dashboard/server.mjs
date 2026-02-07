import { createServer } from "node:http";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { computeSkillsClusters } from "./lib/skills-clusters.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, "public");

const PORT = Number(process.env.PORT || 5177);
const HOST = process.env.HOST || "127.0.0.1";

// WORKSPACE_DIR 자동 탐색 로직
// 1. 환경변수 우선
// 2. server.mjs가 있는 폴더의 상위 폴더 (ghostchat/dashboard의 상위)
// 3. openclaw-main 또는 workspace 폴더가 있는 곳
// 4. process.cwd() (현재 작업 디렉토리)
function findWorkspaceDir() {
  if (process.env.OPENCLAW_WORKSPACE) {
    return process.env.OPENCLAW_WORKSPACE;
  }
  
  // server.mjs가 있는 폴더의 상위 폴더
  const serverDir = path.dirname(fileURLToPath(import.meta.url));
  const parentDir = path.join(serverDir, "..");
  
  // openclaw-main 또는 workspace 폴더가 있는지 확인
  if (existsSync(path.join(parentDir, "openclaw-main", "openclaw.mjs"))) {
    return parentDir;
  }
  if (existsSync(path.join(parentDir, "workspace"))) {
    return parentDir;
  }
  
  // 현재 작업 디렉토리 확인
  const cwd = process.cwd();
  if (existsSync(path.join(cwd, "openclaw-main", "openclaw.mjs"))) {
    return cwd;
  }
  if (existsSync(path.join(cwd, "workspace"))) {
    return cwd;
  }
  
  // 기본값: server.mjs의 상위 폴더
  return parentDir;
}

const WORKSPACE_DIR = findWorkspaceDir();

function resolveCliCommand() {
  if (process.env.OPENCLAW_CLI_ARGS) {
    try {
      const parsed = JSON.parse(process.env.OPENCLAW_CLI_ARGS);
      if (!Array.isArray(parsed) || parsed.length === 0) {
        throw new Error("OPENCLAW_CLI_ARGS must be a non-empty JSON array");
      }
      return { cmd: String(parsed[0]), args: parsed.slice(1).map(String) };
    } catch (err) {
      throw new Error(
        `Invalid OPENCLAW_CLI_ARGS: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }

  const cliEnv = process.env.OPENCLAW_CLI?.trim();
  if (cliEnv) {
    if (cliEnv.endsWith(".mjs") || cliEnv.endsWith(".js")) {
      return { cmd: process.execPath, args: [cliEnv] };
    }
    return { cmd: cliEnv, args: [] };
  }

  // WORKSPACE_DIR 기준으로 openclaw-main/openclaw.mjs 탐색
  const localCli = path.join(WORKSPACE_DIR, "openclaw-main", "openclaw.mjs");
  if (existsSync(localCli)) {
    return { cmd: process.execPath, args: [localCli] };
  }
  
  // 상위 폴더들 탐색 (최대 3단계)
  let searchDir = WORKSPACE_DIR;
  for (let i = 0; i < 3; i++) {
    const candidate = path.join(searchDir, "openclaw-main", "openclaw.mjs");
    if (existsSync(candidate)) {
      return { cmd: process.execPath, args: [candidate] };
    }
    const parent = path.dirname(searchDir);
    if (parent === searchDir) break; // 루트에 도달
    searchDir = parent;
  }
  
  // PATH에서 openclaw 찾기
  return { cmd: "openclaw", args: [] };
}

function runCli(extraArgs) {
  const { cmd, args } = resolveCliCommand();
  const result = spawnSync(cmd, [...args, ...extraArgs], {
    cwd: WORKSPACE_DIR,
    encoding: "utf8",
    maxBuffer: 50 * 1024 * 1024,
  });
  if (result.error) {
    throw new Error(result.error.message);
  }
  if (result.status !== 0) {
    const err = result.stderr?.trim() || result.stdout?.trim() || "Unknown error";
    throw new Error(err);
  }
  return result.stdout ?? "";
}

function resolveClawhubCommand() {
  const envCli = (process.env.CLAWHUB_CLI || process.env.CLAW_HUB_CLI || "").trim();
  if (envCli) {
    const envCheck = spawnSync(envCli, ["-V"], { encoding: "utf8", timeout: 3000 });
    if (!envCheck.error && envCheck.status === 0) {
      return { cmd: envCli, args: [] };
    }
  }

  const candidates = [];
  const appData = process.env.APPDATA;
  const localAppData = process.env.LOCALAPPDATA;
  const programFiles = process.env.ProgramFiles;
  const programFilesX86 = process.env["ProgramFiles(x86)"];

  if (appData) candidates.push(path.join(appData, "npm", "clawhub.cmd"));
  if (localAppData) candidates.push(path.join(localAppData, "npm", "clawhub.cmd"));
  if (programFiles) candidates.push(path.join(programFiles, "nodejs", "clawhub.cmd"));
  if (programFilesX86) candidates.push(path.join(programFilesX86, "nodejs", "clawhub.cmd"));

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      const ext = path.extname(candidate).toLowerCase();
      const candidateCheck =
        ext === ".cmd" || ext === ".bat"
          ? spawnSync("cmd", ["/c", candidate, "-V"], { encoding: "utf8", timeout: 3000 })
          : spawnSync(candidate, ["-V"], { encoding: "utf8", timeout: 3000 });
      if (!candidateCheck.error && candidateCheck.status === 0) {
        if (ext === ".cmd" || ext === ".bat") {
          return { cmd: "cmd", args: ["/c", candidate] };
        }
        return { cmd: candidate, args: [] };
      }
    }
  }

  // Try direct PATH
  const check = spawnSync("clawhub", ["-V"], { encoding: "utf8", timeout: 3000 });
  if (!check.error && check.status === 0) {
    return { cmd: "clawhub", args: [] };
  }
  // Windows: try `where clawhub` to get absolute path
  try {
    const whereResult = spawnSync("where", ["clawhub"], { encoding: "utf8", timeout: 3000 });
    if (!whereResult.error && whereResult.status === 0) {
      const first = (whereResult.stdout || "").split(/\r?\n/).find((l) => l.trim());
      if (first) {
        const abs = first.trim();
        const ext = path.extname(abs).toLowerCase();
        const absCheck =
          ext === ".cmd" || ext === ".bat"
            ? spawnSync("cmd", ["/c", abs, "-V"], { encoding: "utf8", timeout: 3000 })
            : spawnSync(abs, ["-V"], { encoding: "utf8", timeout: 3000 });
        if (!absCheck.error && absCheck.status === 0) {
          if (ext === ".cmd" || ext === ".bat") {
            return { cmd: "cmd", args: ["/c", abs] };
          }
          return { cmd: abs, args: [] };
        }
      }
    }
  } catch {}
  const npxCheck = spawnSync("npx", ["--yes", "clawhub", "-V"], {
    encoding: "utf8",
    timeout: 8000,
  });
  if (!npxCheck.error && npxCheck.status === 0) {
    return { cmd: "npx", args: ["--yes", "clawhub"] };
  }
  const npmExecCheck = spawnSync("npm", ["exec", "--yes", "clawhub", "-V"], {
    encoding: "utf8",
    timeout: 8000,
  });
  if (!npmExecCheck.error && npmExecCheck.status === 0) {
    return { cmd: "npm", args: ["exec", "--yes", "clawhub"] };
  }
  return null;
}

function clawhubDiagnostics() {
  const diag = {
    path: process.env.PATH || "",
    which: "",
    npx: "",
    npm: "",
    candidates: [],
    resolved: null,
  };
  const appData = process.env.APPDATA;
  const localAppData = process.env.LOCALAPPDATA;
  const programFiles = process.env.ProgramFiles;
  const programFilesX86 = process.env["ProgramFiles(x86)"];
  if (appData) diag.candidates.push(path.join(appData, "npm", "clawhub.cmd"));
  if (localAppData) diag.candidates.push(path.join(localAppData, "npm", "clawhub.cmd"));
  if (programFiles) diag.candidates.push(path.join(programFiles, "nodejs", "clawhub.cmd"));
  if (programFilesX86) diag.candidates.push(path.join(programFilesX86, "nodejs", "clawhub.cmd"));
  try {
    diag.resolved = resolveClawhubCommand();
  } catch {
    diag.resolved = null;
  }
  try {
    const whereResult = spawnSync("where", ["clawhub"], { encoding: "utf8", timeout: 3000 });
    diag.which = (whereResult.stdout || whereResult.stderr || "").trim();
  } catch {}
  try {
    const npxCheck = spawnSync("npx", ["--yes", "clawhub", "-V"], {
      encoding: "utf8",
      timeout: 8000,
    });
    diag.npx = (npxCheck.stdout || npxCheck.stderr || "").trim();
  } catch {}
  try {
    const npmExecCheck = spawnSync("npm", ["exec", "--yes", "clawhub", "-V"], {
      encoding: "utf8",
      timeout: 8000,
    });
    diag.npm = (npmExecCheck.stdout || npmExecCheck.stderr || "").trim();
  } catch {}
  return diag;
}

function parseJsonFromCli(output) {
  if (!output || typeof output !== "string") {
    throw new Error("CLI output is not a string");
  }
  
  const trimmed = output.trim();
  
  // "Not found" 같은 단순 텍스트 메시지 체크
  if (trimmed === "Not found" || trimmed === "not found" || trimmed.toLowerCase() === "not found") {
    throw new Error("Not found");
  }
  
  // JSON 배열 또는 객체 찾기
  const jsonStart = output.indexOf("[");
  const objStart = output.indexOf("{");
  
  let start = -1;
  if (jsonStart >= 0 && (objStart < 0 || jsonStart < objStart)) {
    start = jsonStart;
  } else if (objStart >= 0) {
    start = objStart;
  }
  
  if (start < 0) {
    // JSON이 없으면 에러 메시지 추출 시도
    if (trimmed.toLowerCase().includes("not found") || trimmed.toLowerCase().includes("error")) {
      throw new Error(trimmed);
    }
    throw new Error(`CLI did not return JSON. Output: ${trimmed.substring(0, 100)}`);
  }
  
  try {
    return JSON.parse(output.slice(start));
  } catch (parseErr) {
    throw new Error(`Failed to parse JSON: ${parseErr.message}. Output: ${output.substring(start, start + 200)}`);
  }
}

// ClawHub CLI 텍스트 출력 파싱 (search/explore 명령어)
// 형식: "slug v1.0.0  Description  (score)" 또는 "slug  v0.1.0  4m ago  Description..."
function parseClawHubTextOutput(output) {
  if (!output || typeof output !== "string") {
    return [];
  }
  
  const lines = output.trim().split("\n").filter(line => {
    const trimmed = line.trim();
    // 헤더 라인 제외 ("- Searching", "- Fetching latest skills" 등)
    return trimmed && !trimmed.startsWith("-") && trimmed.length > 0;
  });
  
  const results = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    
    // 실제 출력 형식:
    // "slug  v버전  시간  설명..."
    // "slug  v버전  시간" (설명 없음)
    // "slug v버전  Description  (score)" (검색 결과)
    
    // 형식 1: "slug v1.0.0  Description  (score)" - 검색 결과
    const match1 = trimmed.match(/^([^\s]+)\s+v?([\d.]+)\s+(.+?)\s+\(([\d.]+)\)$/);
    if (match1) {
      const [, slug, version, description, score] = match1;
      results.push({
        slug: slug.trim(),
        name: slug.trim(),
        version: version.trim(),
        description: description.trim(),
        score: parseFloat(score),
      });
      continue;
    }
    
    // 형식 2 & 3: "slug  v버전  시간 [설명...]" - explore 결과
    // 시간 형식: "just now", "1m ago", "10m ago", "1h ago", "2m ago" 등
    // 설명이 있을 수도 있고 없을 수도 있음
    // 정규식: slug + 버전 + 시간(공백 포함 가능) + 설명(선택)
    const match2 = trimmed.match(/^([^\s]+)\s+v?([\d.]+)\s+((?:just\s+now|\d+\w+\s+ago))(?:\s+(.+))?$/);
    if (match2) {
      const [, slug, version, timeAgo, description] = match2;
      results.push({
        slug: slug.trim(),
        name: slug.trim(),
        version: version.trim(),
        description: (description || "").trim(),
        timeAgo: timeAgo.trim(),
      });
      continue;
    }
    
    // 형식 4: 간단한 파싱 시도 (공백으로 분리) - explore 형식
    // "slug  v버전  시간  설명..."
    const parts = trimmed.split(/\s+/);
    if (parts.length >= 2) {
      const slug = parts[0];
      let version = parts[1];
      if (version.startsWith("v")) {
        version = version.substring(1);
      }
      
      // 나머지가 시간인지 확인
      let description = "";
      let timeAgo = "";
      if (parts.length >= 3) {
        const thirdPart = parts[2];
        const fourthPart = parts[3];
        
        // "just now" 형식
        if (thirdPart === "just" && fourthPart === "now") {
          timeAgo = "just now";
          description = parts.slice(4).join(" ").trim();
        }
        // "1m ago", "10m ago" 형식
        else if (thirdPart.match(/^\d+\w+$/) && fourthPart === "ago") {
          timeAgo = `${thirdPart} ago`;
          description = parts.slice(4).join(" ").trim();
        }
        // "just"만 있고 "now"가 없거나 다른 형식
        else if (thirdPart === "just") {
          timeAgo = "just now";
          description = parts.slice(3).join(" ").trim();
        }
        // 숫자+문자만 있는 경우 (예: "2m")
        else if (thirdPart.match(/^\d+\w+$/)) {
          // 다음이 "ago"인지 확인
          if (fourthPart === "ago") {
            timeAgo = `${thirdPart} ago`;
            description = parts.slice(4).join(" ").trim();
          } else {
            // 시간이 아닐 수도 있음, 그냥 설명으로 처리
            description = parts.slice(2).join(" ").replace(/\s*\([\d.]+\).*$/, "").trim();
          }
        }
        // 그 외는 모두 설명
        else {
          description = parts.slice(2).join(" ").replace(/\s*\([\d.]+\).*$/, "").trim();
        }
      }
      
      if (slug && version) {
        results.push({
          slug: slug,
          name: slug,
          version: version,
          description: description || "",
          timeAgo: timeAgo || undefined,
        });
      }
    }
  }
  
  return results;
}

function jsonResponse(res, status, payload) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function textResponse(res, status, body, type = "text/plain") {
  res.writeHead(status, {
    "Content-Type": `${type}; charset=utf-8`,
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
      if (data.length > 1_000_000) {
        reject(new Error("Payload too large"));
        req.destroy();
      }
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

function serveStatic(req, res) {
  const urlPath = new URL(req.url, `http://${req.headers.host}`).pathname;
  const relative = urlPath === "/" ? "/index.html" : urlPath;
  const filePath = path.normalize(path.join(publicDir, relative));
  if (!filePath.startsWith(publicDir)) {
    return textResponse(res, 403, "Forbidden");
  }
  if (!existsSync(filePath)) {
    return textResponse(res, 404, "Not found");
  }
  const ext = path.extname(filePath).toLowerCase();
  const type =
    ext === ".html"
      ? "text/html"
      : ext === ".js"
        ? "text/javascript"
        : ext === ".css"
          ? "text/css"
          : "application/octet-stream";
  const body = readFileSync(filePath);
  res.writeHead(200, {
    "Content-Type": `${type}; charset=utf-8`,
    "Content-Length": body.length,
    "Cache-Control": "no-store",
  });
  res.end(body);
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === "/api/skills" && req.method === "GET") {
      try {
        const output = runCli(["skills", "list", "--json"]);
        const data = parseJsonFromCli(output);
        return jsonResponse(res, 200, {
          generatedAt: new Date().toISOString(),
          workspaceDir: data.workspaceDir,
          managedSkillsDir: data.managedSkillsDir,
          skills: data.skills ?? [],
        });
      } catch (err) {
        // 스킬 목록 조회 실패 시에도 빈 목록 반환 (게이트웨이 없어도 스킬 목록은 볼 수 있음)
        return jsonResponse(res, 200, {
          generatedAt: new Date().toISOString(),
          workspaceDir: null,
          managedSkillsDir: null,
          skills: [],
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (url.pathname === "/api/info" && req.method === "GET") {
      return jsonResponse(res, 200, {
        name: "ghostchat-skills-dashboard",
        version: "1.0.0",
        workspaceDir: WORKSPACE_DIR,
        cli: resolveCliCommand(),
      });
    }

    if (url.pathname === "/api/clusters" && req.method === "GET") {
      try {
        const output = runCli(["skills", "list", "--json"]);
        const data = parseJsonFromCli(output);
        const clusters = computeSkillsClusters(data);
        return jsonResponse(res, 200, clusters);
      } catch (err) {
        // 클러스터 조회 실패 시에도 빈 클러스터 반환
        return jsonResponse(res, 200, {
          byStatus: {},
          byMissingType: {},
          bySource: {},
        });
      }
    }

    if (url.pathname === "/api/skills/toggle" && req.method === "POST") {
      const raw = await readBody(req);
      const payload = raw ? JSON.parse(raw) : {};
      const name = String(payload?.name ?? "").trim();
      const enabled = payload?.enabled;
      if (!name || typeof enabled !== "boolean") {
        return jsonResponse(res, 400, { error: "name and enabled(boolean) required" });
      }

      const infoOut = runCli(["skills", "info", "--json", name]);
      const info = parseJsonFromCli(infoOut);
      const skillKey = info.skillKey ?? name;

      runCli(["config", "set", `skills.entries.${skillKey}.enabled`, String(enabled)]);

      const updatedOut = runCli(["skills", "info", "--json", name]);
      const updated = parseJsonFromCli(updatedOut);
      return jsonResponse(res, 200, {
        ok: true,
        restartRequired: true,
        skill: updated,
      });
    }

    if (url.pathname === "/api/gateway/status" && req.method === "GET") {
      try {
        const result = spawnSync(process.execPath, [
          path.join(WORKSPACE_DIR, "openclaw-main", "openclaw.mjs"),
          "gateway",
          "probe",
          "--port",
          "18789",
        ], {
          cwd: WORKSPACE_DIR,
          encoding: "utf8",
          timeout: 5000,
        });
        const isRunning = result.status === 0;
        return jsonResponse(res, 200, {
          running: isRunning,
          port: 18789,
        });
      } catch (err) {
        return jsonResponse(res, 200, {
          running: false,
          port: 18789,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (url.pathname === "/api/install-bin" && req.method === "POST") {
      try {
        const raw = await readBody(req);
        const payload = raw ? JSON.parse(raw) : {};
        const { bin, installCmd } = payload;
        
        if (!bin || !installCmd) {
          return jsonResponse(res, 400, { error: "bin and installCmd required" });
        }
        
        // Windows에서 새 터미널 창 열어서 설치 명령 실행
        const { platform } = require("process");
        if (platform === "win32") {
          const { exec } = require("child_process");
          // PowerShell 새 창에서 실행
          const psCmd = `Start-Process powershell -ArgumentList "-NoExit", "-Command", "${installCmd.replace(/"/g, '\\"')}"`;
          exec(psCmd, (error) => {
            if (error) {
              return jsonResponse(res, 500, { error: error.message });
            }
            return jsonResponse(res, 200, {
              ok: true,
              message: `Installation command opened in new terminal: ${installCmd}`,
            });
          });
          return; // 비동기 응답
        } else {
          return jsonResponse(res, 400, { error: "Auto-install only supported on Windows" });
        }
      } catch (err) {
        return jsonResponse(res, 500, {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (url.pathname === "/api/gateway/restart" && req.method === "POST") {
      try {
        // 게이트웨이 재시작 (SIGUSR1 또는 gateway restart)
        const result = spawnSync(process.execPath, [
          path.join(WORKSPACE_DIR, "openclaw-main", "openclaw.mjs"),
          "gateway",
          "restart",
        ], {
          cwd: WORKSPACE_DIR,
          encoding: "utf8",
          timeout: 10000,
        });
        
        if (result.status === 0) {
          return jsonResponse(res, 200, {
            ok: true,
            message: "Gateway restart initiated",
          });
        } else {
          const error = result.stderr?.trim() || result.stdout?.trim() || "Unknown error";
          return jsonResponse(res, 500, {
            ok: false,
            error: error,
          });
        }
      } catch (err) {
        return jsonResponse(res, 500, {
          ok: false,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    // ClawHub API 엔드포인트
    if (url.pathname === "/api/clawhub/search" && req.method === "GET") {
      try {
        const query = url.searchParams.get("q") || "";
        const page = parseInt(url.searchParams.get("page") || "1", 10);
        const limit = parseInt(url.searchParams.get("limit") || "20", 10);
        
        // 빈 검색어일 경우 인기 스킬 목록 조회 (여러 키워드로 검색하여 합치기)
        let allResults = [];
        
        if (!query || query.trim() === "") {
          const clawhubCmd = resolveClawhubCommand();
          if (!clawhubCmd) {
            return jsonResponse(res, 503, {
              error: "ClawHub CLI not found. Install with: npm i -g clawhub",
              needsInstall: true,
              diagnostics: clawhubDiagnostics(),
              results: [],
              total: 0,
              page: 1,
              limit: limit,
            });
          }
          
          // 빈 검색어일 경우 explore 명령어로 최신 스킬 목록 가져오기
          try {
            // ClawHub CLI는 stdout이 아닌 stderr로 출력할 수 있음
            const exploreResult = spawnSync(
              clawhubCmd.cmd,
              [...clawhubCmd.args, "explore", "--limit", "100"],
              {
                cwd: WORKSPACE_DIR,
                encoding: "utf8",
                timeout: 15000,
                stdio: ["ignore", "pipe", "pipe"], // stdin 무시, stdout/stderr 모두 캡처
              },
            );
            
            if (exploreResult.error?.code === "ENOENT") {
              return jsonResponse(res, 503, {
                error: "ClawHub CLI not found. Install with: npm i -g clawhub",
                needsInstall: true,
                diagnostics: clawhubDiagnostics(),
                results: [],
                total: 0,
                page: 1,
                limit: limit,
              });
            }
            
            if (exploreResult.status === 0 || exploreResult.status === null) {
              // ClawHub CLI는 stdout 또는 stderr로 출력할 수 있음
              const stdout = exploreResult.stdout?.trim() || "";
              const stderr = exploreResult.stderr?.trim() || "";
              // 둘 중 하나에 출력이 있으면 사용
              const output = stdout || stderr;
              
              // "Not found" 같은 에러 메시지 체크
              if (output.toLowerCase().includes("not found")) {
                console.warn("ClawHub explore returned 'Not found'");
                allResults = [];
              } else if (output.toLowerCase().includes("error") && !output.includes("Fetching")) {
                console.warn("ClawHub explore error in output:", output.substring(0, 200));
                allResults = [];
              } else if (output) {
                allResults = parseClawHubTextOutput(output);
                const lineCount = output.split('\n').filter(l => l.trim() && !l.trim().startsWith('-')).length;
                console.log(`ClawHub explore: parsed ${allResults.length} skills from ${lineCount} lines (stdout: ${stdout.length}, stderr: ${stderr.length})`);
                if (allResults.length === 0 && output.includes("Fetching")) {
                  console.warn("ClawHub explore: output exists but parsed 0 skills. First 500 chars:", output.substring(0, 500));
                  console.warn("Full output lines:", output.split('\n').slice(0, 10));
                }
              } else {
                console.warn("ClawHub explore: no output (stdout empty, stderr empty)");
                allResults = [];
              }
            } else {
              // 명령어 실행 실패
              const errorMsg = exploreResult.stderr?.trim() || exploreResult.stdout?.trim() || "Unknown error";
              console.warn(`ClawHub explore failed (status ${exploreResult.status}):`, errorMsg);
              allResults = [];
            }
          } catch (err) {
            console.warn("ClawHub explore failed:", err.message);
            allResults = [];
          }
        } else {
          const clawhubCmd = resolveClawhubCommand();
          if (!clawhubCmd) {
            return jsonResponse(res, 503, {
              error: "ClawHub CLI not found. Install with: npm i -g clawhub",
              needsInstall: true,
              diagnostics: clawhubDiagnostics(),
              results: [],
              total: 0,
              page: 1,
              limit: limit,
            });
          }
          
          // 검색어가 있으면 검색 실행 (--json 옵션 없이)
          const result = spawnSync(
            clawhubCmd.cmd,
            [...clawhubCmd.args, "search", query, "--limit", "100"],
            {
            cwd: WORKSPACE_DIR,
            encoding: "utf8",
            timeout: 15000,
            },
          );
          
          if (result.error?.code === "ENOENT") {
            return jsonResponse(res, 503, {
              error: "ClawHub CLI not found. Install with: npm i -g clawhub",
              needsInstall: true,
              diagnostics: clawhubDiagnostics(),
              results: [],
              total: 0,
              page: 1,
              limit: limit,
            });
          }
          
          if (result.status !== 0) {
            const error = result.stderr?.trim() || result.stdout?.trim() || "Unknown error";
            // "Not found" 같은 메시지는 정상적인 경우일 수 있음
            if (error.toLowerCase().includes("not found")) {
              console.warn("ClawHub search returned 'Not found'");
              allResults = [];
            } else {
              console.warn(`ClawHub search failed (status ${result.status}):`, error);
              return jsonResponse(res, 500, {
                error: error,
                results: [],
                total: 0,
                page: 1,
                limit: limit,
              });
            }
          } else {
            // 성공한 경우 텍스트 출력 파싱
            // ClawHub CLI는 stdout 또는 stderr로 출력할 수 있음
            const stdout = result.stdout?.trim() || "";
            const stderr = result.stderr?.trim() || "";
            const output = stdout || stderr;
            
            if (output.toLowerCase().includes("not found")) {
              console.warn("ClawHub search returned 'Not found'");
              allResults = [];
            } else if (output) {
              allResults = parseClawHubTextOutput(output);
              console.log(`ClawHub search: parsed ${allResults.length} skills (stdout: ${stdout.length}, stderr: ${stderr.length})`);
            } else {
              console.warn("ClawHub search: no output (stdout empty, stderr empty)");
              allResults = [];
            }
          }
        }
        
        // 페이징 처리
        const total = allResults.length;
        const startIndex = (page - 1) * limit;
        const endIndex = startIndex + limit;
        const paginatedResults = allResults.slice(startIndex, endIndex);
        
        return jsonResponse(res, 200, {
          query: query,
          results: paginatedResults,
          total: total,
          page: page,
          limit: limit,
          totalPages: Math.ceil(total / limit),
        });
      } catch (err) {
        return jsonResponse(res, 500, {
          error: err instanceof Error ? err.message : String(err),
          results: [],
          total: 0,
          page: 1,
          limit: 20,
        });
      }
    }

    if (url.pathname === "/api/clawhub/list" && req.method === "GET") {
      try {
        // clawhub list 실행 (설치된 스킬 목록)
        const clawhubCmd = resolveClawhubCommand();
        if (!clawhubCmd) {
          return jsonResponse(res, 503, {
            error: "ClawHub CLI not found. Install with: npm i -g clawhub",
            needsInstall: true,
            diagnostics: clawhubDiagnostics(),
            skills: [],
          });
        }
        const result = spawnSync(clawhubCmd.cmd, [...clawhubCmd.args, "list", "--json"], {
          cwd: WORKSPACE_DIR,
          encoding: "utf8",
          timeout: 10000,
        });
        
        if (result.status !== 0) {
          if (result.error?.code === "ENOENT") {
            return jsonResponse(res, 503, {
              error: "ClawHub CLI not found. Install with: npm i -g clawhub",
              needsInstall: true,
              diagnostics: clawhubDiagnostics(),
              skills: [],
            });
          }
          // list가 비어있을 수도 있음 (에러가 아닐 수 있음)
          return jsonResponse(res, 200, {
            skills: [],
            error: result.stderr?.trim() || undefined,
          });
        }
        
        try {
          const data = parseJsonFromCli(result.stdout);
          return jsonResponse(res, 200, {
            skills: Array.isArray(data) ? data : (data.skills || []),
          });
        } catch (parseErr) {
          return jsonResponse(res, 200, {
            skills: [],
            raw: result.stdout,
          });
        }
      } catch (err) {
        return jsonResponse(res, 500, {
          error: err instanceof Error ? err.message : String(err),
          skills: [],
        });
      }
    }

    if (url.pathname === "/api/clawhub/install" && req.method === "POST") {
      try {
        const raw = await readBody(req);
        const payload = raw ? JSON.parse(raw) : {};
        const slug = String(payload?.slug ?? "").trim();
        
        if (!slug) {
          return jsonResponse(res, 400, { error: "slug required" });
        }
        
        // clawhub install 실행
        const clawhubCmd = resolveClawhubCommand();
        if (!clawhubCmd) {
          return jsonResponse(res, 503, {
            error: "ClawHub CLI not found. Install with: npm i -g clawhub",
            needsInstall: true,
            diagnostics: clawhubDiagnostics(),
          });
        }
        const result = spawnSync(clawhubCmd.cmd, [...clawhubCmd.args, "install", slug], {
          cwd: WORKSPACE_DIR,
          encoding: "utf8",
          timeout: 60000, // 설치에는 시간이 걸릴 수 있음
        });
        
        if (result.status !== 0) {
          if (result.error?.code === "ENOENT") {
            return jsonResponse(res, 503, {
              error: "ClawHub CLI not found. Install with: npm i -g clawhub",
              needsInstall: true,
              diagnostics: clawhubDiagnostics(),
            });
          }
          const error = result.stderr?.trim() || result.stdout?.trim() || "Unknown error";
          return jsonResponse(res, 500, {
            error: error,
            stdout: result.stdout?.trim(),
          });
        }
        
        return jsonResponse(res, 200, {
          ok: true,
          message: `Skill "${slug}" installed successfully`,
          stdout: result.stdout?.trim(),
        });
      } catch (err) {
        return jsonResponse(res, 500, {
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }

    if (req.method === "GET") {
      return serveStatic(req, res);
    }

    return textResponse(res, 405, "Method not allowed");
  } catch (err) {
    return jsonResponse(res, 500, {
      error: err instanceof Error ? err.message : String(err),
      hint:
        "Set OPENCLAW_CLI or OPENCLAW_CLI_ARGS if the OpenClaw CLI is not on PATH. Set OPENCLAW_WORKSPACE to your project root.",
    });
  }
});

server.listen(PORT, HOST, () => {
  // eslint-disable-next-line no-console
  console.log(`OpenClaw Skills Dashboard running at http://${HOST}:${PORT}`);
});
