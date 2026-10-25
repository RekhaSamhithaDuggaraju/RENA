import { Verdict, VERDICT_COLOR } from "@/lib/types";

export function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) {
    return (
      <span className="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-gray-100 text-gray-500">
        Pending
      </span>
    );
  }
  const color = VERDICT_COLOR[verdict];
  return (
    <span
      className="inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold"
      style={{ backgroundColor: `${color}1A`, color }}
    >
      {verdict}
    </span>
  );
}
