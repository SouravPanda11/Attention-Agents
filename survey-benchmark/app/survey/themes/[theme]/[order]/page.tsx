import { notFound } from "next/navigation";
import { SurveyRunner } from "@/components/SurveyRunner";
import { buildThemeWorkflow } from "@/lib/benchmark/themeWorkflow";
import { THEME_IDS, isThemeId } from "@/lib/benchmark/questions/themes/types";
import { ORDER_IDS, isOrderId } from "@/lib/benchmark/schema";

export const dynamicParams = false;

export function generateStaticParams() {
  return THEME_IDS.flatMap((theme) => ORDER_IDS.map((order) => ({ theme, order })));
}

export default async function ThemePage({ params }: { params: Promise<{ theme: string; order: string }> }) {
  const { theme, order } = await params;
  if (!isThemeId(theme) || !isOrderId(order)) notFound();
  return <SurveyRunner workflow={buildThemeWorkflow(theme, order)} />;
}
