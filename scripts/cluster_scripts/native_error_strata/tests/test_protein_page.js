// Runs the page's own script against a stub DOM, exercises every view, and prints values to compare.
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const script = html.slice(html.lastIndexOf("<script>") + 8, html.lastIndexOf("</script>"));

function stub() {
  const node = { value: "", textContent: "", innerHTML: "", hidden: false, style: {}, listeners: {}, nodes: [] };
  node.children = [{ textContent: "" }, { textContent: "" }];
  node.addEventListener = (type, fn) => { node.listeners[type] = fn; };
  node.setAttribute = (key, value) => { node[key] = value; };
  node.appendChild = (child) => { node.nodes.push(child); return child; };
  node.replaceChildren = () => { node.nodes = []; };
  return node;
}
const nodes = {};
globalThis.document = {
  getElementById: (id) => (nodes[id] = nodes[id] || stub()),
  createElement: stub, createElementNS: stub, createTextNode: (text) => ({ text }),
};
globalThis.window = { innerWidth: 1400 };

const checks = `
;(async () => {
  while (!lookup || !heatCells.length) await new Promise((r) => setTimeout(r, 20));
  await new Promise((r) => setTimeout(r, 50));
  const report = (label, value) => console.log(label, value);
  report("proteins", N_P); report("rows", C); report("status", $("loadStatus").textContent.slice(0, 70));
  report("claims rows", DATA.claims.length); report("claims html", $("claimsTable").innerHTML.length);
  const cell = 10 * N_D + 9;
  state.scope = "all"; state.layer = 47; renderAll();
  let r = strata(members(), 47);
  report("all L47 s10 d9", [r.n[cell], r.means[cell].toFixed(4), r.sds[cell].toFixed(4), r.prot[cell]].join(" "));
  r = strata(members(), 0);
  report("all L0 s10 d9", [r.n[cell], r.means[cell].toFixed(4), r.sds[cell].toFixed(4)].join(" "));
  const stz = P.id.indexOf("7stz_C"), c = lookup[stz * N_S * N_D + 12 * N_D + 16];
  report("7stz_C L47 s12 d16", [cellN[c], mean(c, 47).toFixed(4), sd(c, 47).toFixed(4)].join(" "));
  state.scope = "without"; renderAll(); report("without title", $("heatTitle").textContent);
  for (const g of DATA.group_labels.keys()) { state.scope = String(g); renderAll(); report("group", $("heatTitle").textContent + " | " + $("scopeSummary").textContent); }
  selectProtein(stz); report("selected", $("scopeSummary").textContent);
  $("layerRange").listeners.input({ target: { value: "0" } }); report("layer label", $("layerLabel").textContent);
  $("nextProtein").listeners.click(); report("longer", $("protein").value);
  $("previousProtein").listeners.click(); report("shorter", $("protein").value);
  $("protein").listeners.change({ target: { value: "7bew_A" } }); report("typed", $("scopeSummary").textContent);
  $("protein").listeners.change({ target: { get value() { return "nonsense"; }, set value(v) { report("rejected, restored", v); } } });
  heatCells[16 * 1 + 12 * N_D].rect.listeners.click(); report("clicked bin", $("scatterTitle").textContent);
  heatCells[12 * N_D + 16].rect.listeners.mousemove({ clientX: 10, clientY: 10 }); report("tooltip", $("tooltip").nodes.length >= 0 && !$("tooltip").hidden);
  $("seqBin").listeners.change({ target: { value: "0" } }); $("distBin").listeners.change({ target: { value: "16" } });
  report("empty bin dots", $("scatter").nodes.length); report("empty groups", $("groupTable").innerHTML.includes("<td>0</td>"));
  $("seqBin").listeners.change({ target: { value: "12" } }); report("bin tiles", $("binStats").nodes.map((n) => n.children[0].textContent + "=" + n.children[1].textContent).join("; "));
  report("dots", $("scatter").nodes.filter((n) => n.class === "dot").length);
  report("colour 0/+2/-71", [colour(0), colour(2), colour(-71)].map((v) => v.map(Math.round).join(",")).join(" | "));
  console.log("ALL VIEWS RAN");
})().catch((error) => { console.error("FAILED", error); process.exit(1); });
`;
(0, eval)(script + checks);
