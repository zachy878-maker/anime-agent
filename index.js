async function fromJikan() {
  const res = await fetch(
    "https://api.jikan.moe/v4/top/anime?filter=airing&limit=5",
    { headers: { "User-Agent": "anime-agent" } }
  );
  if (!res.ok) throw new Error("Jikan " + res.status);
  const data = await res.json();
  return data.data.map(a => a.title);
}

async function fromKitsu() {
  const res = await fetch(
    "https://kitsu.io/api/edge/trending/anime?limit=5",
    { headers: { "Accept": "application/vnd.api+json" } }
  );
  if (!res.ok) throw new Error("Kitsu " + res.status);
  const data = await res.json();
  return data.data.map(a => a.attributes.canonicalTitle);
}

export default {
  async fetch() {
    const errors = [];
    for (const source of [fromJikan, fromKitsu]) {
      try {
        const titles = await source();
        return new Response(titles.join("\n"));
      } catch (e) {
        errors.push(e.message);
      }
    }
    return new Response("Both failed:\n" + errors.join("\n"));
  }
};
