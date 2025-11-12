#!/usr/bin/env python3
import argparse, os, time, datetime, requests
from dotenv import load_dotenv
import sys, random

# --- config/env ---
load_dotenv()
API_KEY = os.getenv("FINNHUB_TOKEN")  # use FINNHUB_TOKEN from your .env
BASE = "https://finnhub.io/api/v1/quote"

# --- helpers ---
def supports_clear():
    return os.getenv("TERM") not in (None, "dumb")

def header(ts, reason=""):
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== REFRESH @ {stamp} {f'[{reason}]' if reason else ''} ===")

def fetch_quote(symbol: str):
    """Fetch once with error info instead of raising."""
    try:
        r = requests.get(BASE, params={"symbol": symbol, "token": API_KEY}, timeout=10)
        status = r.status_code
        if status == 200:
            j = r.json()
            return {
                "symbol": symbol,
                "price": j.get("c"),
                "chg": j.get("d"),
                "chg%": j.get("dp"),
                "open": j.get("o"),
                "high": j.get("h"),
                "low":  j.get("l"),
                "prev": j.get("pc"),
                "time": datetime.datetime.fromtimestamp(j.get("t", 0)).strftime("%Y-%m-%d %H:%M:%S") if j.get("t") else "-",
            }
        else:
            # common hiccups: 403 (bad/expired token), 429 (rate limit), 5xx
            return {"symbol": symbol, "error": f"HTTP {status}"}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

def print_table(rows):
    headers = ["Symbol","Price","Δ","Δ%","Open","High","Low","Prev Close","As of / Error"]
    widths  = [8,10,10,10,10,10,10,12,22]
    line = "+".join("-"*w for w in widths)
    fmt = " ".join("{:<"+str(w)+"}" for w in widths)
    print(fmt.format(*headers))
    print(line)
    for r in rows:
        chg = r.get("chg")
        color_on = "\033[92m" if isinstance(chg, (int,float)) and chg > 0 else ("\033[91m" if isinstance(chg, (int,float)) and chg < 0 else "")
        color_off = "\033[0m" if color_on else ""
        print(fmt.format(
            r.get("symbol","-"),
            f"{r.get('price'):.2f}" if isinstance(r.get("price"), (int,float)) else "-",
            f"{color_on}{r.get('chg'):+.2f}{color_off}" if isinstance(chg, (int,float)) else "-",
            f"{r.get('chg%'):+.2f}%" if isinstance(r.get("chg%"), (int,float)) else "-",
            f"{r.get('open'):.2f}" if isinstance(r.get("open"), (int,float)) else "-",
            f"{r.get('high'):.2f}" if isinstance(r.get("high"), (int,float)) else "-",
            f"{r.get('low'):.2f}"  if isinstance(r.get("low"),  (int,float)) else "-",
            f"{r.get('prev'):.2f}" if isinstance(r.get("prev"), (int,float)) else "-",
            r.get("time","-") if "error" not in r else r["error"]
        ))

def main():
    ap = argparse.ArgumentParser(description="Simple Finnhub CLI quotes")
    ap.add_argument("symbols", nargs="+", help="Symbols, e.g. AAPL MSFT NVDA")
    ap.add_argument("--watch", type=int, default=0, help="Refresh every N seconds (0 = once)")
    ap.add_argument("--no-clear", action="store_true", help="Do not clear between refreshes")
    args = ap.parse_args()

    if not API_KEY:
        ap.error("FINNHUB_TOKEN not set. Put it in .env (FINNHUB_TOKEN=...) or your environment.")

    try:
        cycle = 0
        while True:
            cycle += 1
            rows = [fetch_quote(s.upper()) for s in args.symbols]
            err = next((r["error"] for r in rows if "error" in r), None)
            reason = err or f"cycle {cycle}"

            if supports_clear() and not args.no_clear:
                os.system("clear")
            header(datetime.datetime.now(), reason)
            printable = []
            for r in rows:
                if "error" in r:
                    printable.append({
                        "symbol": r["symbol"],
                        "price": None, "chg": None, "chg%": None,
                        "open": None, "high": None, "low": None, "prev": None,
                        "time": r["error"]
                    })
                else:
                    printable.append(r)
            print_table(printable)

            if args.watch <= 0:
                break

            # backoff on errors (helps with 429/403)
            wait = args.watch
            if err:
                wait = max(args.watch, 5) + random.uniform(0, 2)

            # visible countdown so you know it’s alive
            for i in range(int(wait), 0, -1):
                print(f"\rNext refresh in {i}s...", end="", flush=True)
                time.sleep(1)
            # handle any fractional remainder from random.uniform
            rem = wait - int(wait)
            if rem > 0:
                time.sleep(rem)
            print("\rRefreshing now...      ")

    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()