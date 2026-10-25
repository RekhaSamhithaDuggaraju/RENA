import { NextRequest, NextResponse } from "next/server";

const protectedPrefixes=["/dashboard","/llm","/mira","/research","/workspace","/attack-lab","/security","/production","/certificates","/audits"];
export function middleware(req:NextRequest){
  const path=req.nextUrl.pathname;
  if(protectedPrefixes.some(prefix=>path===prefix||path.startsWith(prefix+"/")) && !req.cookies.get("rena_token")){
    const url=req.nextUrl.clone(); url.pathname="/"; url.searchParams.set("auth","required"); return NextResponse.redirect(url);
  }
  return NextResponse.next();
}
export const config={matcher:["/dashboard/:path*","/llm/:path*","/mira/:path*","/research/:path*","/workspace/:path*","/attack-lab/:path*","/security/:path*","/production/:path*","/certificates/:path*","/audits/:path*"]};
