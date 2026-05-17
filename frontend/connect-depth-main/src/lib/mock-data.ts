// بيانات تجريبية لـ"بوتيك ليالي" — مشروع فلسطيني للأزياء والستايل

export const business = {
  name: "بوتيك ليالي",
  nameEn: "Layali Boutique",
  handle: "@layali.boutique",
  category: "أزياء وستايل",
  location: "رام الله، فلسطين",
  followers: 48230,
  followersGrowth: 12.4,
  posts: 312,
  avatarColor: "from-fuchsia-500 to-violet-600",
};

export const kpis = [
  { key: "engagement", label: "نسبة التفاعل", value: 7.82, suffix: "%", delta: 14.2, spark: [3,4,4,5,6,5,7,8,7,8] },
  { key: "likes", label: "متوسط اللايكات", value: 2143, delta: 9.1, spark: [1200,1400,1500,1700,1900,1800,2000,2100,2050,2143] },
  { key: "comments", label: "متوسط الكومنتات", value: 184, delta: 22.5, spark: [80,90,110,120,130,140,150,165,170,184] },
  { key: "views", label: "متوسط مشاهدات الريلز", value: 18420, delta: 38.9, spark: [6000,7500,9000,11000,12500,14000,15500,17000,17800,18420] },
  { key: "growth", label: "نمو المتابعين", value: 12.4, suffix: "%", delta: 4.1, spark: [2,3,4,4,5,6,7,8,10,12] },
  { key: "frequency", label: "بوستات بالأسبوع", value: 6.4, delta: -2.0, spark: [7,7,6,7,6,6,7,6,7,6] },
  { key: "besttime", label: "أحسن وقت للنشر", value: "8:30 مساءً", delta: 0, spark: [4,5,6,7,8,9,10,8,7,6], string: true },
  { key: "topcontent", label: "أقوى نوع محتوى", value: "ريلز", delta: 0, spark: [3,4,5,6,7,8,9,10,11,12], string: true },
];

export const engagementOverTime = Array.from({ length: 30 }, (_, i) => {
  const d = new Date(); d.setDate(d.getDate() - (29 - i));
  const months = ["يناير","فبراير","مارس","أبريل","مايو","يونيو","يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"];
  const base = 4 + Math.sin(i / 4) * 1.5 + i * 0.08;
  return {
    date: `${d.getDate()} ${months[d.getMonth()]}`,
    engagement: +(base + Math.random() * 0.8).toFixed(2),
    reach: Math.round(8000 + i * 220 + Math.random() * 1500),
    impressions: Math.round(14000 + i * 380 + Math.random() * 2200),
  };
});

export const heatmapData = (() => {
  const days = ["الأحد","الإثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت"];
  const data: { day: string; hour: number; value: number }[] = [];
  days.forEach((day, di) => {
    for (let h = 0; h < 24; h++) {
      let v = Math.random() * 30;
      if (h >= 19 && h <= 22) v += 50 + Math.random() * 30;
      if (h >= 12 && h <= 14) v += 25;
      if (di === 4 || di === 5) v += 15;
      data.push({ day, hour: h, value: Math.round(v) });
    }
  });
  return data;
})();

export const dayLabels = ["الأحد","الإثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت"];

export const contentTypeData = [
  { name: "ريلز", value: 58, color: "var(--chart-1)" },
  { name: "كاروسيل", value: 22, color: "var(--chart-2)" },
  { name: "صور", value: 14, color: "var(--chart-3)" },
  { name: "ستوري", value: 6, color: "var(--chart-4)" },
];

export const topHashtags = [
  { tag: "#فاشن_فلسطين", uses: 84, eng: 9.2 },
  { tag: "#رام_الله_ستايل", uses: 71, eng: 8.7 },
  { tag: "#أناقة", uses: 65, eng: 7.9 },
  { tag: "#أزياء_فلسطينية", uses: 58, eng: 8.4 },
  { tag: "#إطلالة_اليوم", uses: 52, eng: 6.8 },
  { tag: "#موضة", uses: 49, eng: 7.4 },
  { tag: "#ليالي", uses: 44, eng: 9.6 },
  { tag: "#عبايات", uses: 38, eng: 7.1 },
];

const captions = [
  "إطلالة اليوم بفستان ليالي الجديد ✨ متوفر بألوان الموسم",
  "أجواء خريفية ناعمة 🍂 الكولكشن الجديد بنزّله الليلة الساعة ٨",
  "حقيبة جلد طبيعي بلمسة فلسطينية أصيلة 🇵🇸",
  "من كواليس جلسة التصوير برام الله 📸",
  "عرض حصري لمتابعينا — خصم ٢٠٪ على كل العبايات هاد الأسبوع",
  "إطلالة سهرة 💜 اللينك بالبايو",
  "كولكشن الأعراس وصل — احجزي قبل ما يخلص",
  "٣ طرق تنسّقي فيها سكارف الكتان حقّنا",
  "صباح الخير 🤍 قهوة وستايل وبس",
  "ريل: ٥ إطلالات بأقل من ٢٠٠ شيكل — أيا وحدة بتعجبك؟",
  "كولكشن العيد قريب كتير ✨",
  "شكراً من القلب 💌 لكل وحدة دعمتنا",
];

export const posts = Array.from({ length: 24 }, (_, i) => {
  const types = ["reel", "carousel", "image"] as const;
  const typeLabels: Record<string, string> = { reel: "ريل", carousel: "كاروسيل", image: "صورة" };
  const type = types[i % 3];
  const sponsored = i % 7 === 0;
  const likes = 800 + Math.round(Math.random() * 5000);
  const comments = 30 + Math.round(Math.random() * 400);
  const views = type === "reel" ? 5000 + Math.round(Math.random() * 50000) : 0;
  const sentiments = ["إيجابي", "محايد", "إيجابي", "إيجابي", "سلبي"] as const;
  const themes = [
    ["أزياء", "إطلالة"], ["عرض", "خصم"], ["كواليس"],
    ["أعراس", "فخامة"], ["ستايل", "قهوة"], ["ترفيه", "ريلز"],
    ["ديني", "عيد"], ["زبائن", "آراء"],
  ];
  return {
    id: `p${i + 1}`,
    type,
    typeLabel: typeLabels[type],
    caption: captions[i % captions.length],
    likes, comments, views,
    engagement: +((likes + comments * 4) / 320).toFixed(2),
    sentiment: sentiments[i % sentiments.length],
    themes: themes[i % themes.length],
    hashtags: topHashtags.slice(i % 5, (i % 5) + 3).map(h => h.tag),
    sponsored,
    postedAt: new Date(Date.now() - i * 86400000 * 1.4).toISOString(),
    hue: (i * 47) % 360,
  };
});

export const recommendations = [
  { id: 1, title: "انشري الريلز بين ٧ و ٩ مساءً", impact: "+42% تفاعل", confidence: 94, why: "الريلز اللي نزلتيها بهاد الوقت جابت مشاهدات أعلى ٢.١ مرة من المتوسط اليومي بآخر ٣٠ يوم.", category: "وقت النشر" },
  { id: 2, title: "استخدمي كابشن بالعربي للمحتوى الموضة", impact: "+27% وصول", confidence: 88, why: "البوستات اللي عليها كابشن عربي بتوصل لـ٣١٪ ناس أكتر من جمهورك المحلي برام الله.", category: "محتوى" },
  { id: 3, title: "حطّي ٤–٦ هاشتاجات بس، مش ١٢", impact: "+18% حفظ", confidence: 81, why: "بعد ٧ هاشتاجات بتقل الفايدة — أحسن بوستاتك متوسطها كان ٥.٢ هاشتاج.", category: "هاشتاجات" },
  { id: 4, title: "جرّبي قصص بالكاروسيل للعروض", impact: "+33% حفظ", confidence: 76, why: "الكاروسيل اللي فيه ٥ شرائح أو أكتر بيخلّي الناس تضل تتفرّج ٢.٤ مرة أطول بمشاريع شبيهة.", category: "صيغة" },
  { id: 5, title: "خلّي طول الكابشن بين ٨٠ و ١٤٠ حرف", impact: "+12% كومنتات", confidence: 72, why: "أعلى ٢٠ بوست عندك كان طول الكابشن بهاد المدى. الكابشن الطويل نزّل الكومنتات ١٩٪.", category: "محتوى" },
  { id: 6, title: "استخدمي زر 'اطلبي هلأ' يوم الجمعة", impact: "+24% ضغطات لينك", confidence: 69, why: "الجمعة مساءً مع زر CTA مباشر جابت أعلى تحويلات بالربع الماضي.", category: "زر دعوة" },
];

export const clusters = [
  { id: 1, name: "عروض الأزياء الفاخرة", count: 42, eng: 9.2, color: "oklch(0.55 0.22 277)" },
  { id: 2, name: "ريلز ستايل بسيط", count: 68, eng: 8.4, color: "oklch(0.62 0.2 245)" },
  { id: 3, name: "محتوى ديني موسمي", count: 28, eng: 11.1, color: "oklch(0.7 0.18 195)" },
  { id: 4, name: "ريلز ترفيه وتفاعل", count: 19, eng: 12.7, color: "oklch(0.72 0.18 80)" },
  { id: 5, name: "أزياء الأعراس والمناسبات", count: 35, eng: 10.3, color: "oklch(0.65 0.22 15)" },
  { id: 6, name: "كواليس العمل", count: 24, eng: 7.6, color: "oklch(0.6 0.18 320)" },
];

export const scatterPoints = clusters.flatMap((c, ci) =>
  Array.from({ length: c.count > 40 ? 30 : 20 }, () => {
    const cx = [25, 65, 30, 75, 50, 20][ci];
    const cy = [30, 35, 70, 75, 50, 60][ci];
    return {
      x: cx + (Math.random() - 0.5) * 18,
      y: cy + (Math.random() - 0.5) * 18,
      cluster: c.id,
      name: c.name,
      color: c.color,
    };
  })
);

export const benchmark = {
  percentile: 82,
  sector: "أزياء — فلسطين والشام",
  metrics: [
    { label: "التفاعل", you: 7.8, peer: 4.2, top: 9.6 },
    { label: "وتيرة النشر", you: 6.4, peer: 4.8, top: 8.1 },
    { label: "أداء الريلز", you: 18420, peer: 9200, top: 24500 },
    { label: "وصول الهاشتاجات", you: 84, peer: 52, top: 110 },
    { label: "نسبة الكومنتات", you: 1.8, peer: 0.9, top: 2.4 },
    { label: "نسبة الحفظ", you: 3.1, peer: 1.6, top: 4.2 },
  ],
  radar: [
    { metric: "التفاعل", you: 90, peer: 50 },
    { metric: "الوصول", you: 78, peer: 55 },
    { metric: "الوتيرة", you: 82, peer: 60 },
    { metric: "الجودة", you: 88, peer: 62 },
    { metric: "الهاشتاجات", you: 74, peer: 48 },
    { metric: "الجمهور", you: 80, peer: 58 },
  ],
  leaderboard: [
    { name: "بوتيك ليالي", score: 92, you: true },
    { name: "نور أتلييه", score: 88 },
    { name: "ياسمين ستايل", score: 85 },
    { name: "بيت الطرز", score: 81 },
    { name: "ريم فاشن هاوس", score: 77 },
  ],
};

export const forecast = Array.from({ length: 60 }, (_, i) => {
  const isFuture = i >= 30;
  const base = 4500 + i * 130 + Math.sin(i / 5) * 600;
  return {
    day: `يوم ${i + 1}`,
    actual: isFuture ? null : Math.round(base + Math.random() * 400),
    predicted: isFuture ? Math.round(base) : null,
    upper: isFuture ? Math.round(base + 800 + (i - 30) * 40) : null,
    lower: isFuture ? Math.round(base - 800 - (i - 30) * 40) : null,
  };
});

export const hashtagNetwork = (() => {
  const tags = topHashtags.map((h, i) => ({
    id: h.tag,
    label: h.tag,
    size: 14 + h.uses / 6,
    eng: h.eng,
    group: i % 3,
    x: 50 + Math.cos((i / topHashtags.length) * Math.PI * 2) * 32,
    y: 50 + Math.sin((i / topHashtags.length) * Math.PI * 2) * 32,
  }));
  const links: { source: string; target: string; w: number }[] = [];
  for (let i = 0; i < tags.length; i++) {
    for (let j = i + 1; j < tags.length; j++) {
      if (Math.random() > 0.55) links.push({ source: tags[i].id, target: tags[j].id, w: Math.random() });
    }
  }
  return { nodes: tags, links };
})();

export const insightStream = [
  "الريلز اللي بتنزل بعد الساعة ٧ مساءً أداؤها أحسن بـ٤٢٪ من بوستات الصبح",
  "ريلز الموضة اللي عليها كابشن عربي تفاعلها أعلى بـ٣١٪",
  "الكاروسيل أداؤه أقل من الصور المفردة بـ١٨٪ بعطلة الأسبوع",
  "ريلز الترفيه وقت المشاهدة فيها ٢.٤ مرة أطول من ريلز العروض",
  "أقوى هاشتاج عندك #فاشن_فلسطين جاب ٢٨٪ من وصول الشهر الماضي",
  "بوستات الأربعاء الساعة ٨:٣٠ مساءً دايماً بتفوق على باقي الأوقات",
  "المحتوى المموّل أداؤه أقل من العضوي بـ١٤٪ — جرّبي ستايل طبيعي",
  "المحتوى الديني الموسمي بيوصل لقمته قبل العيد بـ٩ أيام",
];
