import { Link, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  BarChart3,
  FileText,
  Hash,
  LayoutDashboard,
  Layers,
  Settings,
  Sparkles,
  Target,
  TrendingUp,
  Users,
  Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { business as fallbackBusiness } from "@/lib/mock-data";
import { fetchMeData, getAvatarLetter, resolveDatasetId, type MeBusiness } from "@/lib/me-api";

const items = [
  { to: "/dashboard", label: "نظرة عامة", icon: LayoutDashboard },
  { to: "/dashboard/content", label: "المحتوى", icon: BarChart3 },
  { to: "/dashboard/recommendations", label: "التوصيات", icon: Sparkles },
  { to: "/dashboard/hashtags", label: "تجميع المحتوى", icon: Hash },
  { to: "/dashboard/similar", label: "شركات مشابهة", icon: Users },
  { to: "/dashboard/clustering", label: "شبكة واستفسارات", icon: Layers },
  { to: "/dashboard/benchmarking", label: "التوقعات", icon: Target },
  { to: "/dashboard/forecasting", label: "السيناريو", icon: TrendingUp },
  { to: "/dashboard/reports", label: "التقارير", icon: FileText },
  { to: "/dashboard/recommendation-apis-single", label: "API: التوصيات", icon: Sparkles },
  { to: "/dashboard/forecast-single", label: "API: التوقعات", icon: TrendingUp },
  { to: "/dashboard/anomalies-single", label: "API: الشذوذ", icon: Activity },
  { to: "/dashboard/business-momentum-single", label: "API: الزخم", icon: Target },
];

export function Sidebar() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const [business, setBusiness] = useState<MeBusiness>(fallbackBusiness);

  useEffect(() => {
    const datasetId = resolveDatasetId();
    fetchMeData(datasetId)
      .then((data) => setBusiness(data.business))
      .catch(() => {
        setBusiness(fallbackBusiness);
      });
  }, []);

  return (
    <aside className="hidden md:flex w-64 shrink-0 flex-col border-l border-sidebar-border bg-sidebar">
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-sidebar-border">
        <div className="relative h-9 w-9 rounded-xl bg-gradient-brand flex items-center justify-center shadow-glow">
          <Zap className="h-4.5 w-4.5 text-white" strokeWidth={2.5} />
        </div>
        <div>
          <div className="font-display font-semibold text-sm tracking-tight">Suhail</div>
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">ذكاء اجتماعي</div>
        </div>
      </div>

      <div className="px-3 py-4">
        <Link to="/dashboard" className="flex items-center gap-3 p-2.5 rounded-xl hover:bg-sidebar-accent transition-colors">
          <div className={`h-9 w-9 rounded-lg bg-gradient-to-br ${business.avatarColor} flex items-center justify-center text-white text-xs font-semibold`}>{getAvatarLetter(business.name)}</div>
          <div className="min-w-0">
            <div className="text-sm font-medium truncate">{business.name}</div>
            <div className="text-xs text-muted-foreground truncate">{business.handle}</div>
          </div>
        </Link>
      </div>

      <nav className="px-3 flex-1 space-y-0.5">
        <div className="px-3 py-2 text-[10px] uppercase tracking-widest text-muted-foreground">تحليلات</div>
        {items.map((it) => {
          const active = path === it.to || (it.to !== "/dashboard" && path.startsWith(it.to));
          return (
            <Link
              key={it.to}
              to={it.to}
              className={`relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all ${
                active
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                  : "text-sidebar-foreground/70 hover:text-sidebar-foreground hover:bg-sidebar-accent/50"
              }`}
            >
              {active && <motion.span layoutId="activePill" className="absolute right-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-l-full bg-gradient-brand" />}
              <it.icon className="h-4 w-4" />
              <span>{it.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-3 border-t border-sidebar-border">
        <Link to="/dashboard" className="flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-sidebar-foreground/70 hover:bg-sidebar-accent">
          <Settings className="h-4 w-4" /> الإعدادات
        </Link>
      </div>
    </aside>
  );
}
