import { createFileRoute, Outlet, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";

const titles: Record<string, { t: string; s: string }> = {
  "/dashboard": { t: "نظرة عامة", s: "ذكاء اجتماعي فوري لنشاطك التجاري" },
  "/dashboard/content": { t: "تحليل المحتوى", s: "تحليل أداء المنشورات والريلز والقصص" },
  "/dashboard/recommendations": { t: "التوصيات", s: "إجراءات مرتبة حسب التأثير المتوقع" },
  "/dashboard/hashtags": { t: "تجميع المحتوى", s: "ارفع ملفًا وشغّل توصيات الهاشتاقات" },
  "/dashboard/similar": { t: "شركات مشابهة", s: "رؤى مستوحاة من أنشطة قريبة من ملفك" },
  "/dashboard/clustering": { t: "شبكة واستفسارات", s: "اكتشف الأنماط المخفية عبر محتواك" },
  "/dashboard/benchmarking": { t: "التوقعات", s: "قارن مؤشراتك بقطاعك" },
  "/dashboard/forecasting": { t: "السيناريو", s: "توقعات التفاعل ونمو المتابعين" },
  "/dashboard/reports": { t: "التقارير", s: "ملخصات جاهزة للإدارة والتصدير" },
  "/dashboard/recommendation-apis-single": { t: "API التوصيات", s: "تشغيل وعرض كامل استجابات recommendation-apis-single" },
  "/dashboard/forecast-single": { t: "API التوقعات", s: "تشغيل وعرض كامل استجابات forecast-single" },
  "/dashboard/anomalies-single": { t: "API الشذوذ", s: "تشغيل وعرض كامل استجابات anomalies-single" },
  "/dashboard/business-momentum-single": { t: "API زخم الأعمال", s: "تشغيل وعرض كامل استجابات business-momentum-single" },
};

export const Route = createFileRoute("/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard - Suhail" }] }),
  component: DashboardLayout,
});

function DashboardLayout() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const meta = titles[path] ?? titles["/dashboard"];
  return (
    <div className="min-h-screen flex w-full bg-background">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <Topbar title={meta.t} subtitle={meta.s} />
        <main className="flex-1 p-4 md:p-6 lg:p-8 max-w-[1600px] w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
