export default {
  async fetch() {
    try {
      const res = await fetch(
        "https://api.jikan.moe/v4/top/anime?filter=airing&limit=5",
        { headers: { "User-Agent": "anime-agent" } }
      );
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); }
      catch { return new Response("Jikan replied: " + res.status + "\n" + text.slice(0, 300)); }
      if (!data.data) return new Response("Jikan error:\n" + text.slice(0, 300));
      const titles = data.data.map(a => a.title);
      return new Response(titles.join("\n"));
    } catch (e) {
      return new Response("Error: " + e.message);
    }
  }
};
