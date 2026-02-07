function toArray(value) {
  return Array.isArray(value) ? value : [];
}

function sortSkillsByName(skills) {
  return [...skills].sort((a, b) =>
    String(a?.name ?? "").localeCompare(String(b?.name ?? ""), "en"),
  );
}

function getMissingTypes(skill) {
  const missing = skill?.missing ?? {};
  const types = new Set();

  if (toArray(missing.bins).length) types.add("bins");
  if (toArray(missing.anyBins).length) types.add("anyBins");
  if (toArray(missing.env).length) types.add("env");
  if (toArray(missing.config).length) types.add("config");
  if (toArray(missing.os).length) types.add("os");

  return [...types].sort();
}

function statusOf(skill) {
  if (skill?.disabled) return "disabled";
  if (skill?.blockedByAllowlist) return "blockedByAllowlist";
  if (skill?.eligible) return "eligible";
  return "ineligible";
}

function indexByKey(skills, keyFn) {
  const map = new Map();
  for (const skill of skills) {
    const key = keyFn(skill);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(skill);
  }
  return map;
}

function indexByMany(skills, keysFn) {
  const map = new Map();
  for (const skill of skills) {
    const keys = keysFn(skill);
    for (const key of keys) {
      if (!map.has(key)) map.set(key, []);
      map.get(key).push(skill);
    }
  }
  return map;
}

function mapToObject(map, valueMapper = (skills) => skills) {
  const obj = {};
  for (const [key, value] of map.entries()) obj[key] = valueMapper(value);
  return obj;
}

export function computeSkillsClusters(catalog, opts = {}) {
  const now = opts.generatedAt ?? new Date().toISOString();
  const skills = Array.isArray(catalog?.skills) ? catalog.skills : [];

  const clusters = {
    generatedAt: now,
    counts: {
      total: skills.length,
      eligible: skills.filter((s) => Boolean(s?.eligible)).length,
      disabled: skills.filter((s) => Boolean(s?.disabled)).length,
      blockedByAllowlist: skills.filter((s) => Boolean(s?.blockedByAllowlist)).length,
    },
    byStatus: mapToObject(indexByKey(skills, statusOf), (arr) =>
      sortSkillsByName(arr).map((s) => s.name),
    ),
    bySource: mapToObject(indexByKey(skills, (s) => s?.source ?? "unknown"), (arr) =>
      sortSkillsByName(arr).map((s) => s.name),
    ),
    byMissingType: mapToObject(
      indexByMany(skills, (s) =>
        getMissingTypes(s).length ? getMissingTypes(s) : ["none"],
      ),
      (arr) => sortSkillsByName(arr).map((s) => s.name),
    ),
    byMissing: {
      bins: mapToObject(
        indexByMany(skills, (s) => toArray(s?.missing?.bins)),
        (arr) => sortSkillsByName(arr).map((s) => s.name),
      ),
      anyBins: mapToObject(
        indexByMany(skills, (s) => toArray(s?.missing?.anyBins)),
        (arr) => sortSkillsByName(arr).map((s) => s.name),
      ),
      env: mapToObject(
        indexByMany(skills, (s) => toArray(s?.missing?.env)),
        (arr) => sortSkillsByName(arr).map((s) => s.name),
      ),
      config: mapToObject(
        indexByMany(skills, (s) => toArray(s?.missing?.config)),
        (arr) => sortSkillsByName(arr).map((s) => s.name),
      ),
      os: mapToObject(
        indexByMany(skills, (s) => toArray(s?.missing?.os)),
        (arr) => sortSkillsByName(arr).map((s) => s.name),
      ),
    },
  };

  return clusters;
}
