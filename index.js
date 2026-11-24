export default {
  async fetch() {
    try {
      const query = `{ Page(perPage: 5) {
        media(sort: TRENDING_DESC, type: ANIME) {
          title { romaji }
        } } }`;
      const res = await fetch("https://graphql.anilist.co", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
          "User-Agent": "anime-agent"
        },
        body: JSON.stringify({ query })
      });
      const text = await res.text();
      let data;
      try { data = JSON.parse(text); }
      catch { return new Response("AniList replied: " + res.status + "\n" + text.slice(0, 300)); }
      if (!data.data) return new Response("AniList error:\n" + text.slice(0, 300));
      const titles = data.data.Page.media.map(m => m.title.romaji);
      return new Response(titles.join("\n"));
    } catch (e) {
      return new Response("Error: " + e.message);
    }
  }
};
