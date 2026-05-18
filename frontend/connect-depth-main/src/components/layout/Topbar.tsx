import { Search, Calendar, Filter, Download, Bell, Sparkles, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";

export function Topbar({ title, subtitle }: { title: string; subtitle?: string }) {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return (
    <header className="sticky top-0 z-30 h-16 border-b border-border glass flex items-center px-4 md:px-6 gap-3">
      <div className="flex-1 min-w-0">
        <h1 className="text-base md:text-lg font-display font-semibold tracking-tight truncate">{title}</h1>
        {subtitle && <p className="text-xs text-muted-foreground truncate">{subtitle}</p>}
      </div>

      <div className="hidden lg:flex items-center gap-2 px-3 h-9 rounded-lg bg-muted/60 border border-border/60 w-72">
        <Search className="h-4 w-4 text-muted-foreground" />
        <input placeholder="Search posts, hashtags, or insights..." className="bg-transparent outline-none text-sm flex-1 placeholder:text-muted-foreground" />
        <kbd className="text-[10px] text-muted-foreground border border-border rounded px-1.5">Ctrl+K</kbd>
      </div>

      <Button variant="outline" size="sm" className="hidden md:inline-flex h-9 gap-2"><Calendar className="h-3.5 w-3.5" />Last 30 Days</Button>
      <Button variant="outline" size="sm" className="hidden md:inline-flex h-9 gap-2"><Filter className="h-3.5 w-3.5" />Instagram</Button>
      <Button size="sm" className="h-9 gap-2 bg-gradient-brand text-white hover:opacity-90 border-0"><Sparkles className="h-3.5 w-3.5" />Ask AI</Button>

      <div className="h-6 w-px bg-border" />
      <Button variant="ghost" size="icon" className="h-9 w-9" onClick={() => setDark(!dark)}>{dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}</Button>
      <Button variant="ghost" size="icon" className="h-9 w-9 relative"><Bell className="h-4 w-4" /><span className="absolute top-2 right-2 h-1.5 w-1.5 rounded-full bg-gradient-brand" /></Button>
      <Button variant="outline" size="sm" className="h-9 gap-2"><Download className="h-3.5 w-3.5" />Export</Button>
    </header>
  );
}
