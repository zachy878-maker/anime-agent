export default {
  async fetch() {
    const query = `{ Page(perPage: 5) {
      media(sort: TRENDING_DESC, type: ANIME) {
        title { romaji }
      } } }`;
    const res = await fetch("https://graphql.anilist.co", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    const titles = data.data.Page.media.map(m => m.title.romaji);
    return new Response(titles.join("\n"));
  }
};
