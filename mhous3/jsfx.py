"""A JSFX (REAPER-style) subset: header, sliderN, @init / @slider / @block / @sample, expressions, if / loop / while,
mem[] buffers and the common math functions.  Scripts are parsed and translated to Python source (only whitelisted
constructs, so a script cannot run arbitrary Python), then executed per sample.

Not supported: user-defined functions, imports, @gfx, @serialize, gmem, string functions, FFT helpers."""
import math
import random
import re
from functools import lru_cache

import numpy as np

SR = 48000
MEM_SIZE = 1 << 20


class JsfxError(Exception):
    pass


# ----------------------------------------------------------------------------- tokenizer
_TOK = re.compile(r"""\s*(?:(?P<num>0x[0-9a-fA-F]+|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|(?P<id>\$?[A-Za-z_][A-Za-z0-9_.]*)|(?P<op><<|>>|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=|%=|\^=|\|=|&=|[-+*/%^&|~!<>=?:;,()\[\]]))""")


def tokenize(src):
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    pos, out = 0, []
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = _TOK.match(src, pos)
        if not m:
            raise JsfxError(f"Unexpected character near: {src[pos:pos + 20]!r}")
        pos = m.end()
        if m.group("num"):
            t = m.group("num")
            out.append(("num", float(int(t, 16)) if t.lower().startswith("0x") else float(t)))
        elif m.group("id"):
            out.append(("id", m.group("id")))
        else:
            out.append(("op", m.group("op")))
    out.append(("eof", None))
    return out


# ----------------------------------------------------------------------------- parser
FUNCS = {"sin": "math.sin", "cos": "math.cos", "tan": "math.tan", "asin": "math.asin", "acos": "math.acos", "atan": "math.atan",
         "atan2": "math.atan2", "sqrt": "_sqrt", "exp": "math.exp", "log": "_log", "log10": "_log10", "pow": "_pow", "abs": "abs",
         "min": "min", "max": "max", "sign": "_sign", "floor": "math.floor", "ceil": "math.ceil", "int": "math.trunc", "rand": "_rand",
         "sqr": "_sqr", "fmod": "math.fmod", "tanh": "math.tanh"}
CONSTS = {"$pi": math.pi, "$e": math.e, "$phi": (1 + 5 ** 0.5) / 2}


class Parser:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i]

    def eat(self, kind=None, val=None):
        tok = self.t[self.i]
        if (kind and tok[0] != kind) or (val is not None and tok[1] != val):
            raise JsfxError(f"Expected {val or kind} but found {tok[1]!r}")
        self.i += 1
        return tok

    def is_op(self, v):
        return self.t[self.i] == ("op", v)

    def statements(self, until=None):
        out = []
        while self.peek()[0] != "eof" and not (until and self.is_op(until)):
            if self.is_op(";"):
                self.i += 1
                continue
            out.append(self.expr())
            if self.is_op(";"):
                self.i += 1
            elif not (until and self.is_op(until)) and self.peek()[0] != "eof":
                raise JsfxError(f"Missing ';' before {self.peek()[1]!r}")
        return out

    def expr(self):
        return self.assign()

    def assign(self):
        lhs = self.ternary()
        tok = self.peek()
        if tok[0] == "op" and tok[1] in ("=", "+=", "-=", "*=", "/=", "%=", "^=", "|=", "&="):
            if lhs[0] not in ("var", "idx"):
                raise JsfxError("Left side of an assignment must be a variable or buffer element")
            self.i += 1
            return ("assign", tok[1], lhs, self.assign())
        return lhs

    def ternary(self):
        c = self.binary(0)
        if self.is_op("?"):
            self.i += 1
            a = self.assign()
            b = ("num", 0.0)
            if self.is_op(":"):
                self.i += 1
                b = self.assign()
            return ("tern", c, a, b)
        return c

    LEVELS = [("||",), ("&&",), ("|",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"), ("+", "-"), ("*", "/", "%")]

    def binary(self, lvl):
        if lvl == len(self.LEVELS):
            return self.power()
        left = self.binary(lvl + 1)
        while self.peek()[0] == "op" and self.peek()[1] in self.LEVELS[lvl]:
            op = self.eat()[1]
            left = ("bin", op, left, self.binary(lvl + 1))
        return left

    def power(self):
        base = self.unary()
        if self.is_op("^"):
            self.i += 1
            return ("bin", "^", base, self.power())
        return base

    def unary(self):
        if self.peek()[0] == "op" and self.peek()[1] in ("-", "+", "!", "~"):
            op = self.eat()[1]
            return ("un", op, self.unary())
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok[0] == "num":
            self.i += 1
            return ("num", tok[1])
        if tok == ("op", "("):
            self.i += 1
            stmts = self.statements(")")
            self.eat("op", ")")
            return stmts[0] if len(stmts) == 1 else ("block", stmts)
        if tok[0] == "id":
            self.i += 1
            name = tok[1]
            if name in CONSTS:
                return ("num", CONSTS[name])
            if name in ("if", "while") and self.is_op("("):
                self.i += 1
                cond = self.expr()
                self.eat("op", ")")
                body = self.primary() if self.is_op("(") else self.expr()
                other = None
                if name == "if" and self.peek() == ("id", "else"):
                    self.i += 1
                    other = self.primary() if self.is_op("(") else self.expr()
                return (name, cond, body, other)
            if name == "loop" and self.is_op("("):
                self.i += 1
                n = self.expr()
                self.eat("op", ",")
                stmts = self.statements(")")
                self.eat("op", ")")
                return ("loop", n, stmts[0] if len(stmts) == 1 else ("block", stmts))
            if self.is_op("("):
                if name not in FUNCS:
                    raise JsfxError(f"Unsupported function: {name}()")
                self.i += 1
                args = []
                while not self.is_op(")"):
                    args.append(self.expr())
                    if self.is_op(","):
                        self.i += 1
                self.eat("op", ")")
                return ("call", name, args)
            if self.is_op("["):
                self.i += 1
                idx = self.expr()
                self.eat("op", "]")
                return ("idx", name, idx)
            return ("var", name)
        raise JsfxError(f"Unexpected token {tok[1]!r}")


# ----------------------------------------------------------------------------- code generation
def _vn(name):
    return "v_" + name.replace(".", "__").replace("$", "S_")


class Gen:
    def __init__(self):
        self.names = set()

    def var(self, name):
        self.names.add(name)
        return _vn(name)

    def truth(self, e):
        return f"(({self.expr(e)}) != 0.0)"

    def expr(self, n):
        k = n[0]
        if k == "num":
            return repr(n[1])
        if k == "var":
            return self.var(n[1])
        if k == "idx":
            return f"mem[_ix({self.var(n[1])} + {self.expr(n[2])})]"
        if k == "call":
            return f"float({FUNCS[n[1]]}({', '.join(self.expr(a) for a in n[2])}))"
        if k == "un":
            e = self.expr(n[2])
            return {"-": f"(-{e})", "+": f"(+{e})", "!": f"(0.0 if {e} != 0.0 else 1.0)", "~": f"float(~int({e}))"}[n[1]]
        if k == "bin":
            a, b, op = self.expr(n[2]), self.expr(n[3]), n[1]
            if op in ("+", "-", "*"):
                return f"({a} {op} {b})"
            if op == "/":
                return f"_div({a}, {b})"
            if op == "%":
                return f"_mod({a}, {b})"
            if op == "^":
                return f"_pow({a}, {b})"
            if op in ("<", ">", "<=", ">=", "==", "!="):
                return f"(1.0 if {a} {op} {b} else 0.0)"
            if op == "&&":
                return f"(1.0 if ({a} != 0.0 and {b} != 0.0) else 0.0)"
            if op == "||":
                return f"(1.0 if ({a} != 0.0 or {b} != 0.0) else 0.0)"
            return f"float(int({a}) {op if op in ('&', '|', '<<', '>>') else op} int({b}))"
        if k == "tern":
            return f"({self.expr(n[2])} if {self.expr(n[1])} != 0.0 else {self.expr(n[3])})"
        if k == "assign":
            if n[2][0] != "var":
                raise JsfxError("Assigning to a buffer element inside an expression is not supported; use it as its own statement")
            v = self.var(n[2][1])
            rhs = self.expr(n[3])
            return f"({v} := {rhs if n[1] == '=' else self._op(v, n[1][:-1], rhs)})"
        if k == "block":
            if all(s[0] == "var" or s[0] == "num" or s[0] in ("bin", "un", "call", "tern", "idx") for s in n[1]):
                return self.expr(n[1][-1])
            raise JsfxError("Multi-statement blocks are only supported as statements (inside if / loop / while)")
        raise JsfxError(f"Cannot use '{k}' inside an expression")

    def _op(self, a, op, b):
        return {"+": f"({a} + {b})", "-": f"({a} - {b})", "*": f"({a} * {b})", "/": f"_div({a}, {b})", "%": f"_mod({a}, {b})",
                "^": f"_pow({a}, {b})", "|": f"float(int({a}) | int({b}))", "&": f"float(int({a}) & int({b}))"}[op]

    def stmt(self, n, ind):
        pad = "    " * ind
        k = n[0]
        if k == "assign":
            tgt = n[2]
            lhs = self.var(tgt[1]) if tgt[0] == "var" else f"mem[_ix({self.var(tgt[1])} + {self.expr(tgt[2])})]"
            rhs = self.expr(n[3])
            return [f"{pad}{lhs} = {rhs if n[1] == '=' else self._op(lhs, n[1][:-1], rhs)}"]
        if k == "block":
            out = []
            for s in n[1]:
                out += self.stmt(s, ind)
            return out or [pad + "pass"]
        if k in ("if", "while"):
            out = [f"{pad}{k} {self.truth(n[1])}:"] + self.stmt(n[2], ind + 1)
            if k == "if" and n[3] is not None:
                out += [f"{pad}else:"] + self.stmt(n[3], ind + 1)
            if k == "while":
                out.insert(1, f"{pad}    _guard = 0")
            return out
        if k == "loop":
            return [f"{pad}for _ in range(int({self.expr(n[1])})):"] + self.stmt(n[2], ind + 1)
        if k == "tern" and (n[2][0] in ("block", "assign") or n[3][0] in ("block", "assign")):
            return [f"{pad}if {self.truth(n[1])}:"] + self.stmt(n[2], ind + 1) + [f"{pad}else:"] + self.stmt(n[3], ind + 1)
        return [f"{pad}_ = {self.expr(n)}"]


def _helpers():
    def _div(a, b): return a / b if b != 0 else 0.0
    def _mod(a, b): return float(int(a) % int(b)) if int(b) != 0 else 0.0
    def _pow(a, b):
        try:
            r = a ** b
            return r.real if isinstance(r, complex) else float(r)
        except (OverflowError, ZeroDivisionError):
            return 0.0
    def _sqrt(a): return math.sqrt(a) if a > 0 else 0.0
    def _log(a): return math.log(a) if a > 0 else 0.0
    def _log10(a): return math.log10(a) if a > 0 else 0.0
    def _sign(a): return 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
    def _sqr(a): return a * a
    def _rand(a): return random.random() * a
    def _ix(i):
        i = int(i)
        if not 0 <= i < MEM_SIZE:
            raise JsfxError("Buffer index out of range")
        return i
    return dict(math=math, _div=_div, _mod=_mod, _pow=_pow, _sqrt=_sqrt, _log=_log, _log10=_log10, _sign=_sign, _sqr=_sqr, _rand=_rand, _ix=_ix)


_HELP = _helpers()


# ----------------------------------------------------------------------------- program
SECTIONS = ("init", "slider", "block", "sample")


def parse_header(script):
    info = dict(desc="JSFX effect", sliders=[])
    for line in script.splitlines():
        line = line.strip()
        if line.startswith("@"):
            break
        if line.startswith("desc:"):
            info["desc"] = line[5:].strip()
        m = re.match(r"slider(\d+):\s*(?:(\w+)=)?\s*(-?[\d.]+)?\s*(?:<\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*(?:,\s*(-?[\d.]+))?\s*(?:\{([^}]*)\})?\s*>)?\s*(.*)$", line)
        if m:
            idx = int(m.group(1))
            default = float(m.group(3) or 0)
            lo = float(m.group(4)) if m.group(4) else 0.0
            hi = float(m.group(5)) if m.group(5) else max(1.0, default)
            step = float(m.group(6)) if m.group(6) else 0.01
            opts = [o.strip() for o in m.group(7).split(",")] if m.group(7) else None
            info["sliders"].append(dict(idx=idx, default=default, min=lo, max=hi, step=step or 0.01, label=m.group(8).strip() or f"slider{idx}", options=opts))
    return info


def _split(script):
    secs, cur = {s: [] for s in SECTIONS}, None
    for line in script.splitlines():
        m = re.match(r"\s*@(\w+)", line)
        if m:
            cur = m.group(1) if m.group(1) in SECTIONS else None
            continue
        if cur:
            secs[cur].append(line)
    return {k: "\n".join(v) for k, v in secs.items()}


@lru_cache(maxsize=32)
def compile_script(script):
    secs = _split(script)
    gen = Gen()
    bodies = {}
    for name in SECTIONS:
        src = secs[name].strip()
        if not src:
            bodies[name] = None
            continue
        try:
            nodes = Parser(tokenize(src)).statements()
        except JsfxError as e:
            raise JsfxError(f"@{name}: {e}")
        lines = []
        for n in nodes:
            lines += gen.stmt(n, 2 if name == "sample" else 1)
        bodies[name] = lines
    for s in ("spl0", "spl1", "srate", "num_ch", "samplesblock", "tempo"):
        gen.names.add(s)
    for i in range(1, 17):
        gen.names.add(f"slider{i}")
    names = sorted(gen.names)
    load = [f"    {_vn(n)} = V.get({n!r}, 0.0)" for n in names]
    store = [f"    V[{n!r}] = {_vn(n)}" for n in names]
    code = []
    for name in ("init", "slider", "block"):
        code += [f"def sec_{name}(V, mem):"] + load + (bodies[name] or ["    pass"]) + store + [""]
    code += ["def sec_sample(V, mem, L, R, N):"] + load
    code += ["    for _i in range(N):", f"        {_vn('spl0')} = L[_i]", f"        {_vn('spl1')} = R[_i]"]
    code += (bodies["sample"] or ["        pass"])
    code += [f"        L[_i] = {_vn('spl0')}", f"        R[_i] = {_vn('spl1')}"] + store + [""]
    ns = dict(_HELP)
    try:
        exec(compile("\n".join(code), "<jsfx>", "exec"), ns)
    except SyntaxError as e:
        raise JsfxError(f"Could not translate the script ({e.msg})")
    return ns, parse_header(script)


def process(x, script, params=None, sr=SR):
    """Run a script over x (n, 2).  params = {'slider1': value, ...} overrides the defaults."""
    ns, info = compile_script(script)
    V = {"srate": float(sr), "num_ch": 2.0, "tempo": 120.0, "samplesblock": 512.0}
    for s in info["sliders"]:
        V[f"slider{s['idx']}"] = float((params or {}).get(f"slider{s['idx']}", s["default"]))
    mem = [0.0] * MEM_SIZE
    ns["sec_init"](V, mem)
    ns["sec_slider"](V, mem)
    L = x[:, 0].astype(np.float64).tolist()
    R = x[:, 1 if x.shape[1] > 1 else 0].astype(np.float64).tolist()
    n, blk = len(L), 512
    try:
        for s in range(0, n, blk):
            e = min(n, s + blk)
            V["samplesblock"] = float(e - s)
            ns["sec_block"](V, mem)
            Lb, Rb = L[s:e], R[s:e]
            ns["sec_sample"](V, mem, Lb, Rb, e - s)
            L[s:e], R[s:e] = Lb, Rb
    except (OverflowError, ZeroDivisionError) as e:
        raise JsfxError(f"Math error while running the script: {e}")
    out = np.stack([np.asarray(L, np.float32), np.asarray(R, np.float32)], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


# ----------------------------------------------------------------------------- examples
EXAMPLES = {
    "Gain (dB)": """desc:Simple gain
slider1:0<-48,24,0.5>Gain (dB)
@slider
g = 10^(slider1/20);
@sample
spl0 *= g; spl1 *= g;
""",
    "Tremolo": """desc:Tremolo
slider1:5<0.1,20,0.1>Rate (Hz)
slider2:0.6<0,1,0.01>Depth
@init
ph = 0;
@sample
ph += 2*$pi*slider1/srate;
ph > 2*$pi ? ph -= 2*$pi;
g = 1 - slider2*(0.5 + 0.5*sin(ph));
spl0 *= g; spl1 *= g;
""",
    "One-pole low-pass": """desc:One-pole low-pass
slider1:3000<100,16000,10>Cutoff (Hz)
@slider
a = exp(-2*$pi*slider1/srate);
@sample
s0 = a*s0 + (1-a)*spl0;
s1 = a*s1 + (1-a)*spl1;
spl0 = s0; spl1 = s1;
""",
    "Lo-fi crusher": """desc:Lo-fi crusher
slider1:6<2,16,1>Bits
slider2:4<1,24,1>Hold (samples)
@init
cnt = 0; hold0 = 0; hold1 = 0;
@sample
cnt >= slider2 ? (cnt = 0; q = 2^(slider1-1); hold0 = floor(spl0*q + 0.5)/q; hold1 = floor(spl1*q + 0.5)/q;);
cnt += 1;
spl0 = hold0; spl1 = hold1;
""",
    "Echo (memory buffer)": """desc:Echo with feedback
slider1:300<20,1000,1>Delay (ms)
slider2:0.4<0,0.9,0.01>Feedback
slider3:0.35<0,1,0.01>Mix
@init
pos = 0;
@slider
len = floor(slider1 * srate / 1000);
@sample
rd = pos - len; rd < 0 ? rd += len;
d0 = buf[rd]; d1 = buf[rd + 100000];
buf[pos] = spl0 + d0*slider2;
buf[pos + 100000] = spl1 + d1*slider2;
spl0 += d0*slider3; spl1 += d1*slider3;
pos += 1; pos >= len ? pos = 0;
""",
}
