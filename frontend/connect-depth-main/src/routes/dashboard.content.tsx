import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { posts } from "@/lib/mock-data";
import { Heart, MessageCircle, Eye, Search, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

const filterLabels: Record<string, string> = { all: "الكل", reel: "ريلز", carousel: "كاروسيل", image: "صور" };

export const Route = createFileRoute("/dashboard/content")({ component: Content });

function Content() {
  const [filter, setFilter] = useState<"all" | "reel" | "carousel" | "image">("all");
  const filtered = posts.filter(p => filter === "all" || p.type === filter);
  return (
    <div className="space-y-5">
      <div className="flex flex-col md:flex-row gap-3 md:items-center justify-between">
        <div className="flex items-center gap-2 px-3 h-10 rounded-xl bg-card border border-border w-full md:w-80">
          <Search className="h-4 w-4 text-muted-foreground" />
          <input placeholder="دوّر بالكابشن أو الهاشتاجات…" className="bg-transparent outline-none text-sm flex-1" />
        </div>
        <div className="flex items-center gap-2">
          {(["all", "reel", "carousel", "image"] as const).map(t => (
            <button key={t} onClick={() => setFilter(t)} className={`h-9 px-3 rounded-lg text-xs font-medium border transition-all ${filter === t ? "bg-gradient-brand text-white border-transparent shadow-glow" : "bg-card border-border text-muted-foreground hover:text-foreground"}`}>{filterLabels[t]}</button>
          ))}
          <button className="h-9 px-3 rounded-lg text-xs font-medium border bg-card border-border flex items-center gap-1.5"><SlidersHorizontal className="h-3.5 w-3.5" />ترتيب: حسب الأداء</button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filtered.map((p, i) => (
          <motion.div key={p.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }} whileHover={{ y: -3 }} className="group rounded-2xl overflow-hidden border border-border bg-card shadow-card hover:shadow-elegant transition-shadow">
            <div className="aspect-[4/5] relative" style={{ background: `linear-gradient(135deg, oklch(0.7 0.18 ${p.hue}), oklch(0.55 0.22 ${(p.hue + 60) % 360}))` }}>
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/10 to-transparent" />
              <div className="absolute top-3 right-3 flex gap-1.5">
                <span className="px-2 py-0.5 text-[10px] uppercase tracking-wider rounded bg-black/40 text-white backdrop-blur-md">{p.typeLabel}</span>
                {p.sponsored && <span className="px-2 py-0.5 text-[10px] tracking-wider rounded bg-warning text-warning-foreground">مموّل</span>}
              </div>
              <div className={`absolute top-3 left-3 px-2 py-0.5 text-[10px] tracking-wider rounded backdrop-blur-md ${p.sentiment === "إيجابي" ? "bg-success/80 text-white" : p.sentiment === "سلبي" ? "bg-destructive/80 text-white" : "bg-muted/80"}`}>{p.sentiment}</div>
              <div className="absolute bottom-0 left-0 right-0 p-3 text-white">
                <p className="text-xs line-clamp-2 leading-relaxed">{p.caption}</p>
              </div>
            </div>
            <div className="p-3 space-y-2.5">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><Heart className="h-3.5 w-3.5" />{p.likes.toLocaleString("ar-EG")}</span>
                <span className="flex items-center gap-1"><MessageCircle className="h-3.5 w-3.5" />{p.comments}</span>
                {p.views > 0 && <span className="flex items-center gap-1"><Eye className="h-3.5 w-3.5" />{(p.views/1000).toFixed(1)} ألف</span>}
              </div>
              <div className="flex items-center justify-between">
                <div className="flex flex-wrap gap-1">
                  {p.themes.slice(0, 2).map(t => <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-accent text-accent-foreground">{t}</span>)}
                </div>
                <div className="flex items-center gap-1">
                  <div className="text-[10px] text-muted-foreground">تفاعل</div>
                  <div className="num-display text-sm font-semibold text-[var(--brand)]">{p.engagement}%</div>
                </div>
              </div>
              <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${Math.min(100, p.engagement * 8)}%` }} transition={{ duration: 1, delay: i * 0.02 }} className="h-full bg-gradient-brand rounded-full" />
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
