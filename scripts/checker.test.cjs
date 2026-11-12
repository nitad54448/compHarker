const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const html=fs.readFileSync(path.join(__dirname,'../check_install_JS.html'),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function harness(resources={}){
  const logs=[],requests=[],nodes=new Map(),node=id=>{if(!nodes.has(id))nodes.set(id,{value:'',appendChild:el=>logs.push(el.textContent)});return nodes.get(id);};
  const ctx=vm.createContext({URL,window:{location:{href:'https://test.local/check_install_JS.html',origin:'https://test.local'}},
    document:{getElementById:node,createElement:()=>({})},fetch:async(url,options)=>{
      requests.push({url,options});return {ok:Object.hasOwn(resources,url),status:Object.hasOwn(resources,url)?200:404,url,
        headers:{get:()=>null},text:async()=>resources[url]};
    }});
  vm.runInContext(script,ctx);return {ctx,logs,requests};
}
test('JavaScript scanner recognizes workers, side-effect imports, re-exports and all importScripts arguments',()=>{
  const h=harness(),warnings=[];
  const refs=h.ctx.extractReferences(`import './setup.js';export {x} from './module.mjs';new Worker('./worker.js?v=2');new SharedWorker(new URL('./shared.mjs',import.meta.url));importScripts('./one.js','./two.js');fetch(new URL('./model.json',import.meta.url));`,'js',warnings);
  assert.deepEqual([...refs].sort(),['./setup.js','./module.mjs','./worker.js?v=2','./shared.mjs','./one.js','./two.js','./model.json'].sort());
  assert.equal(warnings.length,0);
});
test('CSS scanner recognizes imports and quoted or unquoted URLs',()=>{
  const refs=harness().ctx.extractReferences(`@import 'nested.css?v=1';a{background:url("img (1).png")}@font-face{src:url(font.woff2#font)}`,'css');
  assert.deepEqual([...refs],['img (1).png','font.woff2#font','nested.css?v=1']);
  const warnings=[];
  const dynamic=harness().ctx.extractReferences('a{background:url(var(--image))}', 'css', warnings);
  assert.equal(dynamic.length,0);assert(warnings.some(message=>/manual verification/.test(message)));
});
test('HTML scans attributes, inline scripts and inline styles by their resource type',()=>{
  const refs=harness().ctx.extractReferences(`<link href="style.css?x=1"><script src="app.js"></script><script>import './side.js';new Worker('worker.js')</script><style>a{background:url(bg.png)}</style><div style="background:url(icon.svg)"></div>`);
  assert.deepEqual([...refs].sort(),['style.css?x=1','app.js','./side.js','worker.js','bg.png','icon.svg'].sort());
});
test('comments and strings do not become imports; dynamic and bare imports get warnings',()=>{
  const h=harness(),warnings=[];
  const refs=h.ctx.extractReferences(`// new Worker('fake.js')\nconst s="import './fake2.js'";new Worker(path);import 'package';fetch(\`./data/\${name}.json\`);`,'js',warnings);
  assert.equal(refs.length,0);assert.equal(warnings.length,3);
});
test('query strings do not prevent recursion, and fragments do not duplicate checks',async()=>{
  const h=harness({
    'https://test.local/app/worker.mjs?v=1':`import './dependency.js';fetch('./kernel.wgsl')`,
    'https://test.local/app/dependency.js':'',
    'https://test.local/app/kernel.wgsl':'shader',
    'https://test.local/app/style.css?v=2':`@import './nested.css?v=3';a{background:url(icon.svg#one)}`,
    'https://test.local/app/nested.css?v=3':'',
    'https://test.local/app/icon.svg':'svg'
  });
  h.ctx.window.targetUrl='https://test.local/app/index.html';h.ctx.window.mainSource=`<script>new Worker('./worker.mjs?v=1')</script><link href="style.css?v=2"><img src="icon.svg#two">`;
  vm.runInContext('mainFunctionLoaded=true;',h.ctx);await h.ctx.runCheck();
  assert(h.requests.some(r=>r.url.endsWith('dependency.js')));
  assert(h.requests.some(r=>r.url.endsWith('nested.css?v=3')));
  assert.equal(h.requests.filter(r=>r.url.endsWith('icon.svg')).length,1);
  assert.equal(h.requests.find(r=>r.url.endsWith('worker.mjs?v=1')).options.method,'GET');
  assert(!h.logs.some(line=>line.includes('MISSING')));
});
test('unsupported forms are visible in the check output',async()=>{
  const h=harness();h.ctx.window.targetUrl='https://test.local/app.html';h.ctx.window.mainSource='<script>new Worker(workerPath)</script>';
  vm.runInContext('mainFunctionLoaded=true;',h.ctx);await h.ctx.runCheck();assert(h.logs.some(line=>/manual verification/.test(line)));
});
