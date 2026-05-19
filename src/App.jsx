import React, { useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Upload,
  Image as ImageIcon,
  Bot,
  Send,
  Network,
  MapPin,
  Hash,
  Tags,
  GitBranch,
  FileJson,
  Brain,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  Database,
  Eye,
  RefreshCw,
} from "lucide-react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import InteractiveSggGraph from "./InteractiveSggGraph";
const API_BASE = import.meta.env.VITE_API_BASE || "";
const DEMO_IMAGE_NAME = "0069.png";
const DEMO_IMAGE_URL = `${import.meta.env.BASE_URL}demo/0069.png`;

const sampleGraph = {
  objects: [
    { id: "o1", label: "airplane", x: 22, y: 28, box: [18, 23, 30, 35] },
    { id: "o2", label: "airplane", x: 48, y: 34, box: [44, 29, 57, 42] },
    { id: "o3", label: "apron", x: 36, y: 62, box: [15, 51, 68, 77] },
    { id: "o4", label: "taxiway", x: 78, y: 58, box: [67, 52, 92, 65] },
    { id: "o5", label: "boarding_bridge", x: 60, y: 18, box: [54, 13, 68, 24] },
  ],
  relations: [
    { source: "o1", target: "o3", predicate: "parallelly parked on" },
    { source: "o2", target: "o3", predicate: "isolatedly parked on" },
    { source: "o1", target: "o2", predicate: "parked alongside with" },
    { source: "o5", target: "o3", predicate: "over" },
    { source: "o4", target: "o3", predicate: "adjacent" },
  ],
};

const sggQuestions = [
  {
    key: "category",
    title: "场景图类别问题",
    prompt: "What object categories are included in the image?",
    example: "What object categories are included in the image?",
    desc: "What object categories are included in the image?",
    icon: Tags,
  },
  {
    key: "count",
    title: "场景图数量问题",
    prompt: "How many objects of each category are in the image?",
    example: "How many airplanes are in the image?",
    desc: "How many objects of each category are in the image?",
    icon: Hash,
  },
  {
    key: "location",
    title: "场景图空间位置问题",
    prompt: "Where are the objects in the image located?",
    example: "What is the position of the runway in the image?",
    desc: "Where are the objects in the image located?",
    icon: MapPin,
  },
  {
    key: "relationship",
    title: "场景图语义关系问题",
    prompt: "List the relationships of scene graph storage in the image.",
    example: "List the relationships of scene graph storage in the image.",
    desc: "List the relationships of scene graph storage in the image.",
    icon: GitBranch,
  },
];
const defaultJsonQuestion = `What object categories are included in the image?`;

function stripOptionsFromQuestion(text) {
  return String(text || "")
    .replace(/\n\s*A\.\s*[\s\S]*$/i, "")
    .trim();
}

function classNames(...items) {
  return items.filter(Boolean).join(" ");
}

async function postJson(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function postForm(path, formData) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function UploadPanel({ imageUrl, fileName, onUpload, onRunSgg, loading }) {
  const inputRef = useRef(null);

  return (
    <div className="rounded-[2rem] border border-slate-200 bg-white p-5 shadow-xl">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-black uppercase tracking-[0.2em] text-cyan-700">Image Input</p>
          <h2 className="mt-1 text-2xl font-black text-slate-950">任务1：场景图生成</h2>
        </div>
        <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-50 text-cyan-700">
          <ImageIcon className="h-6 w-6" />
        </div>
      </div>

      <button
        onClick={() => inputRef.current?.click()}
        className="group flex min-h-[330px] w-full items-center justify-center overflow-hidden rounded-[1.5rem] border-2 border-dashed border-slate-300 bg-slate-50 transition hover:border-cyan-400 hover:bg-cyan-50/60"
      >
        {imageUrl ? (
          <div className="relative h-full w-full">
            <img src={imageUrl} alt="Uploaded satellite" className="h-[330px] w-full object-cover" />
            <div className="absolute left-4 top-4 rounded-2xl bg-slate-950/75 px-4 py-2 text-sm font-bold text-white backdrop-blur">
              {fileName || "uploaded image"}
            </div>
          </div>
        ) : (
          <div className="px-8 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-3xl bg-white text-cyan-700 shadow-sm transition group-hover:scale-105">
              <Upload className="h-8 w-8" />
            </div>
            <p className="mt-5 text-lg font-black text-slate-900">点击上传卫星图片</p>
            <p className="mt-2 text-sm text-slate-500">支持PNG / JPG / JPEG。上传完成后，请求后端生成SGG检测图和结构图。</p>
          </div>
        )}
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onUpload(file);
        }}
      />

      <div className="mt-4 flex flex-wrap gap-3">
        <button
          onClick={onRunSgg}
          disabled={!imageUrl || loading}
          className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-5 py-3 font-bold text-white shadow-lg transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Network className="h-4 w-4" />}
          生成SGG结构图
        </button>
        <div className="inline-flex items-center gap-2 rounded-2xl bg-slate-100 px-4 py-3 text-sm font-bold text-slate-600">
          <CheckCircle2 className="h-4 w-4 text-emerald-600" /> 场景图生成
        </div>
      </div>
    </div>
  );
}

function ZoomViewerModal({ open, onClose, src, title }) {
  if (!open || !src) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="absolute inset-4 flex flex-col overflow-hidden rounded-[2rem] bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <h3 className="text-xl font-black text-slate-950">{title}</h3>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white"
          >
            关闭
          </button>
        </div>

        <div className="relative flex-1 overflow-hidden bg-slate-100">
          <TransformWrapper
            initialScale={1}
            minScale={0.2}
            maxScale={30}
            wheel={{ step: 0.12 }}
            doubleClick={{ disabled: false }}
            panning={{ disabled: false }}
            centerOnInit
            limitToBounds={false}
          >
            {({ zoomIn, zoomOut, resetTransform }) => (
              <>
                <div className="absolute right-6 top-6 z-10 flex gap-2">
                  <button
                    type="button"
                    onClick={() => zoomIn()}
                    className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white"
                  >
                    放大
                  </button>
                  <button
                    type="button"
                    onClick={() => zoomOut()}
                    className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white"
                  >
                    缩小
                  </button>
                  <button
                    type="button"
                    onClick={() => resetTransform()}
                    className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white"
                  >
                    还原
                  </button>
                </div>

                <TransformComponent
                  wrapperClass="!w-full !h-full"
                  contentClass="!w-full !h-full flex items-center justify-center"
                >
                  <img
                    src={src}
                    alt={title}
                    className="max-h-[80vh] max-w-[90vw] select-none object-contain"
                    draggable={false}
                  />
                </TransformComponent>
              </>
            )}
          </TransformWrapper>
        </div>
      </div>
    </div>
  );
}

function SggGraph({ graph, graphImageUrl, annotatedImageUrl, loading, rawRelationCount, displayRelationCount }) {
  const [zoomData, setZoomData] = useState(null);
  const [graphModalOpen, setGraphModalOpen] = useState(false);

  const hasInteractiveGraph = graph?.relations?.length > 0;

  return (
    <>
      <div className="rounded-[2rem] border border-slate-200 bg-slate-950 p-5 text-white shadow-xl">
        <div className="mb-4 flex items-center justify-between gap-3">
          <div>
            <p className="text-sm font-black uppercase tracking-[0.2em] text-cyan-300">
              SGG Output
            </p>
            <h2 className="mt-1 text-2xl font-black">图像标注 + 结构图</h2>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          {/* 右侧左框：标框图 */}
          <div className="relative min-h-[315px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-white p-3 text-slate-950">
            {annotatedImageUrl ? (
              <button
                type="button"
                onClick={() =>
                  setZoomData({
                    src: annotatedImageUrl,
                    title: "图像标注结果",
                  })
                }
                className="h-full w-full cursor-zoom-in"
              >
                <img
                  src={annotatedImageUrl}
                  alt="SGG annotated detection result"
                  className="h-[315px] w-full object-contain"
                />
              </button>
            ) : (
              <div className="flex h-[315px] w-full items-center justify-center text-slate-400">
                没有标注结果可用
              </div>
            )}

            <div className="absolute bottom-4 left-4 rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-2 text-sm text-slate-200 backdrop-blur">
              {loading ? "Generating detection result..." : "单击可放大/拖动可平移"}
            </div>
          </div>

          {/* 右侧右框：结构图 */}
          <div className="relative min-h-[315px] overflow-hidden rounded-[1.5rem] border border-white/10 bg-white p-3 text-slate-950">
            {hasInteractiveGraph ? (
              <>
                <div className="h-[315px] w-full">
                  <InteractiveSggGraph graph={graph} />
                </div>

                <button
                  type="button"
                  onClick={() => setGraphModalOpen(true)}
                  className="absolute right-4 top-4 rounded-full bg-slate-950/80 px-4 py-2 text-sm font-bold text-white shadow-lg backdrop-blur"
                >
                  放大
                </button>
              </>
            ) : graphImageUrl ? (
              <button
                type="button"
                onClick={() =>
                  setZoomData({
                    src: graphImageUrl,
                    title: "SGG Structure Diagram",
                  })
                }
                className="h-full w-full cursor-zoom-in"
              >
                <img
                  src={graphImageUrl}
                  alt="SGG scene graph"
                  className="h-[315px] w-full object-contain"
                />
              </button>
            ) : (
              <div className="flex h-[315px] w-full items-center justify-center text-slate-400">
                没有结构图结果可用
              </div>
            )}

            <div className="absolute bottom-4 left-4 rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-2 text-sm text-slate-200 backdrop-blur">
              {hasInteractiveGraph
                ? "Drag nodes / blue control points to adjust the structure diagram"
                : "单击可放大/拖动可平移"}
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-sm text-slate-400">Objects</p>
            <p className="mt-1 text-2xl font-black">{graph?.objects?.length || 0}</p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-sm text-slate-400">Unique Relations</p>
            <p className="mt-1 text-2xl font-black">
              {displayRelationCount || graph?.relations?.length || 0}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Raw: {rawRelationCount || graph?.relations?.length || 0}
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
            <p className="text-sm text-slate-400">Mode</p>
            <p className="mt-1 text-lg font-black">STAR-SGG Demo</p>
          </div>
        </div>
      </div>

      <ZoomViewerModal
        open={!!zoomData}
        onClose={() => setZoomData(null)}
        src={zoomData?.src}
        title={zoomData?.title}
      />

      {graphModalOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/80 p-4 backdrop-blur-sm"
          onClick={() => setGraphModalOpen(false)}
        >
          <div
            className="flex h-full flex-col overflow-hidden rounded-[2rem] bg-white shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
              <h3 className="text-xl font-black text-slate-950">SGG交互结构图</h3>

              <button
                type="button"
                onClick={() => setGraphModalOpen(false)}
                className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white"
              >
                关闭
              </button>
            </div>

            <div className="relative min-h-0 flex-1 bg-white">
              <TransformWrapper
                initialScale={1}
                minScale={0.35}
                maxScale={8}
                wheel={{ step: 0.12 }}
                doubleClick={{ disabled: false }}
                panning={{ disabled: false }}
                centerOnInit
                limitToBounds={false}
              >
                {({ zoomIn, zoomOut, resetTransform }) => (
                  <>
                    <div className="absolute right-6 top-6 z-20 flex gap-2">
                      <button
                        type="button"
                        onClick={() => zoomIn()}
                        className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white shadow-lg"
                      >
                        Zoom In +
                      </button>

                      <button
                        type="button"
                        onClick={() => zoomOut()}
                        className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white shadow-lg"
                      >
                        Zoom Out -
                      </button>

                      <button
                        type="button"
                        onClick={() => resetTransform()}
                        className="rounded-full bg-slate-950 px-4 py-2 text-sm font-bold text-white shadow-lg"
                      >
                        Reset
                      </button>
                    </div>

                    <div className="absolute left-6 top-6 z-20 rounded-full bg-slate-950/80 px-4 py-2 text-sm font-bold text-white shadow-lg">
                      轮播缩放 / 拖拽画布 / 拖拽节点和蓝色控制点
                    </div>

                    <TransformComponent
                      wrapperClass="!h-full !w-full"
                      contentClass="!h-full !w-full"
                    >
                      <div className="h-[85vh] w-[1400px] p-6">
                        <InteractiveSggGraph graph={graph} />
                      </div>
                    </TransformComponent>
                  </>
                )}
              </TransformWrapper>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function RobotQaPanel({ answers, activeKey, setActiveKey, onAsk, loadingKey, customQuestion, setCustomQuestion }) {
  const activeQuestion = sggQuestions.find((q) => q.key === activeKey) || sggQuestions[0];
  const handleSubmitCustomQuestion = () => {
    const text = customQuestion.trim();

    if (!text) return;

    onAsk({
      ...activeQuestion,
      prompt: text,
    });
  };

  return (
    <section className="mx-auto w-full max-w-[1600px] px-4 py-8">
      <div className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
        <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-xl">
          <div className="flex items-center gap-4">
            <motion.div
              className="flex h-16 w-16 items-center justify-center rounded-[1.5rem] bg-gradient-to-br from-cyan-400 to-blue-500 text-white shadow-lg shadow-cyan-500/20"
              animate={{ y: [0, -5, 0] }}
              transition={{ duration: 2.2, repeat: Infinity }}
            >
              <Bot className="h-8 w-8" />
            </motion.div>
            <div>
              <p className="text-sm font-black uppercase tracking-[0.2em] text-cyan-700">SGG Assistant</p>
              <h2 className="text-3xl font-black text-slate-950">任务2：场景图问答</h2>
            </div>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {sggQuestions.map((q) => {
              const Icon = q.icon;
              const selected = activeKey === q.key;
              return (
                <button
                  key={q.key}
                  // onClick={() => {setCustomQuestion(q.prompt);onAsk(q);}}
                  onClick={() => { setActiveKey(q.key); setCustomQuestion(q.example || q.prompt); }}
                  className={classNames(
                    "rounded-[1.5rem] border p-5 text-left transition hover:-translate-y-0.5",
                    selected ? "border-cyan-300 bg-cyan-50 shadow-lg shadow-cyan-100" : "border-slate-200 bg-slate-50 hover:bg-white"
                  )}
                >
                  <div className="flex items-start gap-4">
                    <div className={classNames("flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl", selected ? "bg-cyan-500 text-white" : "bg-white text-slate-700")}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-black text-slate-950">{q.title}</h3>
                      <p className="mt-2 text-sm leading-6 text-slate-600">{q.prompt}</p>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="rounded-[2rem] border border-slate-200 bg-slate-950 p-6 text-white shadow-xl">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-slate-950">
                <Bot className="h-6 w-6" />
              </div>
              <div>
                <p className="text-sm text-slate-400">Robot Answer</p>
                <h3 className="text-xl font-black">{activeQuestion.title}</h3>
              </div>
            </div>
            <Sparkles className="h-6 w-6 text-cyan-300" />
          </div>

         <div className="mt-6 rounded-[1.5rem] border border-white/10 bg-white/5 p-5">
            <p className="text-sm font-bold text-cyan-200">Question</p>

            <textarea
              value={customQuestion}
              onChange={(e) => setCustomQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmitCustomQuestion();
                }
              }}
              placeholder="Enter a question, for example: List the relationships stored in the scene graph for this image."
              className="mt-3 min-h-[96px] w-full resize-none rounded-2xl border border-white/10 bg-slate-900/70 p-4 leading-7 text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-cyan-300"
            />

            <div className="mt-3 flex items-center justify-between gap-3">
              <p className="text-xs text-slate-400">按 Enter 键查询。按 Shift + Enter 键换行。</p>

              <button
                type="button"
                onClick={handleSubmitCustomQuestion}
                disabled={loadingKey === activeQuestion.key}
                className="inline-flex items-center gap-2 rounded-xl bg-cyan-400 px-4 py-2 text-sm font-black text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loadingKey === activeQuestion.key ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                问题
              </button>
            </div>
          </div>

          <div className="mt-4 min-h-[190px] rounded-[1.5rem] border border-white/10 bg-white p-5 text-slate-900">
            <div className="mb-3 flex items-center gap-2 text-sm font-black text-slate-500">
              {loadingKey === activeQuestion.key ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4 text-emerald-600" />}
              Answer
            </div>
            <pre className="whitespace-pre-wrap font-sans leading-8">{loadingKey === activeQuestion.key ? "Generating answer..." : answers[activeQuestion.key] || "输入问题并按Enter，或单击Query获取答案。"}</pre>
          </div>
        </div>
      </div>
    </section>
  );
}

function QuestionCard({ title, icon: Icon, accent, children }) {
  return (
    <div className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-xl">
      <div className="mb-5 flex items-center gap-3">
        <div className={classNames("flex h-12 w-12 items-center justify-center rounded-2xl text-white", accent)}>
          <Icon className="h-6 w-6" />
        </div>
        <h2 className="text-2xl font-black text-slate-950">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function BottomQa({ jsonQuestion, setJsonQuestion, jsonAnswer, setJsonAnswer, commonQuestion, setCommonQuestion, commonAnswer, setCommonAnswer, selectedImageName }) {
  const [jsonLoading, setJsonLoading] = useState(false);
  const [commonLoading, setCommonLoading] = useState(false);

  const askJson = async () => {
    setJsonLoading(true);
    try {
      const data = await postJson("/api/kg-rag", {
        question: jsonQuestion,
        image: selectedImageName,
      });

      if (data.question) {
        setJsonQuestion(stripOptionsFromQuestion(data.question));
      }

      setJsonAnswer(data.llm_answer_sentence || data.answer || "");
    } catch (err) {
      console.error("KG-RAG request failed:", err);
      setJsonAnswer(`KG-RAG request failed. Please check the backend at /api/kg-rag.\n\nError message: ${err.message}`);
    } finally {
      setJsonLoading(false);
    }
  };

  const askCommon = async () => {
    setCommonLoading(true);
    try {
      const data = await postJson("/api/common-vqa", {
        question: commonQuestion,
        image: selectedImageName,
      });
      setCommonAnswer(data.answer || JSON.stringify(data, null, 2));
    } catch (err) {
      setCommonAnswer("演示模式回答：飞机、跑道、滑行道和喷气桥在图像中可见。");
    } finally {
      setCommonLoading(false);
    }
  };

  return (
    <section className="mx-auto grid w-full max-w-[1600px] gap-6 px-4 pb-12 pt-4 lg:grid-cols-2">
      <QuestionCard title="任务3：知识图谱问答" icon={FileJson} accent="bg-gradient-to-br from-violet-500 to-fuchsia-500">
        <p className="mb-3 text-sm leading-6 text-slate-600">使用知识图增强LLM推理能力</p>
        <textarea
          value={jsonQuestion}
          onChange={(e) => setJsonQuestion(e.target.value)}
          placeholder="请输入问题"
          className="min-h-[155px] w-full rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-7 outline-none focus:border-violet-400"
        />
        <div className="mt-4 flex items-center gap-3">
          <button onClick={askJson} disabled={jsonLoading} className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-5 py-3 font-bold text-white disabled:opacity-50">
            {jsonLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            获取问题
          </button>
          <span className="text-sm text-slate-500">KGQA</span>
        </div>
        <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="mb-2 text-sm font-black text-slate-500">Output Answer</p>
          <pre className="whitespace-pre-wrap text-sm leading-7 text-slate-800">{jsonAnswer || "等待回答"}</pre>
        </div>
      </QuestionCard>

      <QuestionCard title="任务4：常识问答" icon={Brain} accent="bg-gradient-to-br from-emerald-500 to-teal-500">
        <p className="mb-3 text-sm leading-6 text-slate-600">这是用来问开放式的常识性问题，比如“图片里有什么？”，或者“图片里有什么场景？”
</p>
        <textarea
          value={commonQuestion}
          onChange={(e) => setCommonQuestion(e.target.value)}
          className="min-h-[155px] w-full rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm leading-7 outline-none focus:border-emerald-400"
        />
        <div className="mt-4 flex items-center gap-3">
          <button onClick={askCommon} disabled={commonLoading} className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-5 py-3 font-bold text-white disabled:opacity-50">
            {commonLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />}
            询问常识性问题
          </button>
          <span className="text-sm text-slate-500">连接到LLM</span>
        </div>
        <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="mb-2 text-sm font-black text-slate-500">Output Answer</p>
          <pre className="whitespace-pre-wrap text-sm leading-7 text-slate-800">{commonAnswer || "等待模型输出..."}</pre>
        </div>
      </QuestionCard>
    </section>
  );
}

// const loadRealJsonQuestionByImage = async (imageName) => {
//   if (!imageName) return;

//   setJsonQuestion("The real problem is being read based on the picture...");
//   setJsonAnswer("The real problem is being read based on the picture...");

//   try {
//     const data = await postJson("/api/kg-rag", {
//       image: imageName,
//     });

//     console.log("Auto KG-RAG question response:", data);

//     if (data.status && data.status !== "ok") {
//       setJsonQuestion("The real problem was not found.");
//       setJsonAnswer(data.answer || "The real problem was not found.");
//       return;
//     }

//     // 关键：把后端返回的真实问题放进上面的 textarea
//     setJsonQuestion(stripOptionsFromQuestion(data.question || ""));

//     // 下面只显示答案
//     setJsonAnswer(data.llm_answer_sentence || data.answer || "");
//   } catch (err) {
//     console.error("Auto load KG-RAG question failed:", err);
//     setJsonQuestion("The real problem failed to load.");
//     setJsonAnswer(`Failed to automatically load the real question. Please check the backend /api/kg-rag.\n\nError message: ${err.message}`);
//   }
// };

export default function App() {
  const [imageUrl, setImageUrl] = useState(DEMO_IMAGE_URL);
  const [fileName, setFileName] = useState(DEMO_IMAGE_NAME);
  const [selectedFile, setSelectedFile] = useState(null);
  const [graph, setGraph] = useState({ objects: [], relations: [] });
  const [graphImageUrl, setGraphImageUrl] = useState("");
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState("");
  const [rawRelationCount, setRawRelationCount] = useState(0);
  const [displayRelationCount, setDisplayRelationCount] = useState(0);
  const [sggLoading, setSggLoading] = useState(false);
  const [activeKey, setActiveKey] = useState("category");
  const [answers, setAnswers] = useState({});
  const [loadingKey, setLoadingKey] = useState("");
  const [customQuestion, setCustomQuestion] = useState(sggQuestions[0].prompt);
  const [jsonQuestion, setJsonQuestion] = useState(defaultJsonQuestion);
  const [jsonAnswer, setJsonAnswer] = useState("");
  const [commonQuestion, setCommonQuestion] = useState("图中显示的是什么地方？");
  const [commonAnswer, setCommonAnswer] = useState("");

  const loadRealJsonQuestionByImage = async (imageName) => {
    if (!imageName) return;

    setJsonQuestion("正在根据图像加载真实问题...");
    setJsonAnswer("");

    try {
      const data = await postJson("/api/kg-rag", {
        image: imageName,
      });

      console.log("Auto KG-RAG question response:", data);

      if (data.status && data.status !== "ok") {
        setJsonQuestion("");
        setJsonAnswer(data.answer || "No real question was found for this image.");
        return;
      }

      setJsonQuestion(stripOptionsFromQuestion(data.question || ""));
      setJsonAnswer("");
    } catch (err) {
      console.error("Auto load KG-RAG question failed:", err);
      setJsonQuestion("");
      setJsonAnswer(`Failed to automatically load the real question.\n\nError message: ${err.message}`);
    }
  };

  const handleUpload = (file) => {
    setSelectedFile(file);
    setFileName(file.name);
    setImageUrl(URL.createObjectURL(file));
    setGraph({ objects: [], relations: [] });
    setGraphImageUrl("");
    setAnnotatedImageUrl("");
    setRawRelationCount(0);
    setDisplayRelationCount(0);
    setAnswers({});
    setCommonAnswer("");

    setJsonQuestion("Loading the real question based on the image...");
    setJsonAnswer("");

    loadRealJsonQuestionByImage(file.name);
  };

  const getDemoFile = async () => {
    const res = await fetch(DEMO_IMAGE_URL);

    if (!res.ok) {
      throw new Error(`Failed to load demo image: ${DEMO_IMAGE_URL}`);
    }

    const blob = await res.blob();

    return new File([blob], DEMO_IMAGE_NAME, {
      type: blob.type || "image/png",
    });
  };

  const runSgg = async () => {
    if (!fileName) {
      alert("请先上传或选择一个演示图像。");
      return;
    }

    setSggLoading(true);

    try {
      const fileToSend = selectedFile || (await getDemoFile());

      const formData = new FormData();
      formData.append("image", fileToSend);

      const data = await postForm("/api/sgg", formData);

      console.log("SGG response:", data);

      if (data.status && data.status !== "ok") {
        alert(`SGG generation failed: ${data.message || data.detail || data.status}`);
        return;
      }

      setRawRelationCount(data.rawRelationCount || 0);
      setDisplayRelationCount(data.displayRelationCount || data.graph?.relations?.length || 0);

      if (data.graph) {
        setGraph(data.graph);
      } else {
        setGraph({ objects: [], relations: [] });
      }

      if (data.annotatedImageUrl) {
        const fullAnnotatedUrl = data.annotatedImageUrl.startsWith("http")
          ? data.annotatedImageUrl
          : `${API_BASE}${data.annotatedImageUrl}`;

        setAnnotatedImageUrl(`${fullAnnotatedUrl}?t=${Date.now()}`);
      } else {
        setAnnotatedImageUrl("");
      }

      if (data.graphImageUrl) {
        const fullGraphUrl = data.graphImageUrl.startsWith("http")
          ? data.graphImageUrl
          : `${API_BASE}${data.graphImageUrl}`;

        setGraphImageUrl(`${fullGraphUrl}?t=${Date.now()}`);
      } else {
        setGraphImageUrl("");
      }
    } catch (err) {
      console.error("SGG request failed:", err);
      alert(`SGG generation failed: ${err.message}`);

      setGraph({ objects: [], relations: [] });
      setGraphImageUrl("");
      setAnnotatedImageUrl("");
      setRawRelationCount(0);
      setDisplayRelationCount(0);
    } finally {
      setSggLoading(false);
    }
  };
  const askSggQuestion = async (question) => {
    setActiveKey(question.key);

    const queryText = (question.prompt || customQuestion || "").trim();

    if (!fileName) {
      setAnswers((prev) => ({
        ...prev,
        [question.key]: "Please upload an image first, then select a scene diagram question.",
      }));
      return;
    }

    if (!queryText) {
      setAnswers((prev) => ({
        ...prev,
        [question.key]: "Please enter a question.",
      }));
      return;
    }

    setLoadingKey(question.key);

    try {
      const data = await postJson("/api/sgg-qa", {
        type: question.key,
        question: queryText,
        image: fileName,
      });

      console.log("SGG QA response:", data);

      if (data.status && data.status !== "ok") {
        setAnswers((prev) => ({
          ...prev,
          [question.key]: data.answer || `Scene diagram question answering failed: ${data.status}`,
        }));
        return;
      }

      setAnswers((prev) => ({
        ...prev,
        [question.key]: data.answer || "The backend did not return an answer field.",
      }));
    } catch (err) {
      console.error("SGG QA request failed:", err);

      setAnswers((prev) => ({
        ...prev,
        [question.key]: `Scene diagram question answering request failed. Please check the backend /api/sgg-qa.\n\nError message: ${err.message}`,
      }));
    } finally {
    setLoadingKey("");
    }
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <header className="relative overflow-hidden bg-slate-950 text-white">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_18%_12%,rgba(56,189,248,.24),transparent_32%),radial-gradient(circle_at_82%_18%,rgba(16,185,129,.18),transparent_28%)]" />
        <div className="relative mx-auto max-w-7xl px-6 py-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100">
            <Sparkles className="h-4 w-4" /> STAR-SGG Visual Question Answering Demo
          </div>
          <div className="mt-6 grid gap-8 lg:grid-cols-[1fr_.85fr] lg:items-end">
            <div>
              <h1 className="max-w-4xl text-5xl font-black leading-tight tracking-tight md:text-7xl">
                SGG Image Parsing and <span className="bg-gradient-to-r from-cyan-300 via-emerald-300 to-amber-200 bg-clip-text text-transparent">KG-RAG Question Answering</span>
              </h1>
              <p className="mt-5 max-w-3xl text-lg leading-8 text-slate-300">
                上传遥感图像，显示检测结果和场景图，然后进行机器人问题回答，围绕四类问题：类别、数量、位置和关系。支持知识图谱问答，也支持常识图像问题回答。
              </p>
            </div>
            <div className="rounded-[2rem] border border-white/10 bg-white/5 p-5 backdrop-blur">
              <p className="font-black text-cyan-200">任务总览</p>
              <div className="mt-4 grid gap-2 text-sm text-slate-300">
                <div className="rounded-xl bg-white/5 px-3 py-2">任务1：场景图生成</div>
                <div className="rounded-xl bg-white/5 px-3 py-2">任务2：场景图问答</div>
                <div className="rounded-xl bg-white/5 px-3 py-2">任务3：知识图谱问答</div>
                <div className="rounded-xl bg-white/5 px-3 py-2">任务4：常识问答</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <section className="mx-auto grid w-full max-w-[1600px] gap-6 px-4 py-8 lg:grid-cols-[.85fr_1.15fr]">
        <UploadPanel imageUrl={imageUrl} fileName={fileName} onUpload={handleUpload} onRunSgg={runSgg} loading={sggLoading} />
        <SggGraph graph={graph} graphImageUrl={graphImageUrl} annotatedImageUrl={annotatedImageUrl} loading={sggLoading} rawRelationCount={rawRelationCount} displayRelationCount={displayRelationCount} />
      </section>

      <RobotQaPanel answers={answers} activeKey={activeKey} setActiveKey={setActiveKey} onAsk={askSggQuestion} loadingKey={loadingKey} customQuestion={customQuestion} setCustomQuestion={setCustomQuestion} />

      <BottomQa
        jsonQuestion={jsonQuestion}
        setJsonQuestion={setJsonQuestion}
        jsonAnswer={jsonAnswer}
        setJsonAnswer={setJsonAnswer}
        commonQuestion={commonQuestion}
        setCommonQuestion={setCommonQuestion}
        commonAnswer={commonAnswer}
        setCommonAnswer={setCommonAnswer}
        selectedImageName={fileName}
      />

      <footer className="border-t border-slate-200 bg-white px-6 py-8 text-center text-sm text-slate-500">
        Scene diagram generation demonstration demo.
        Contact: <a href="mailto:dxyufei@lnut.edu.cn" className="text-cyan-500 hover:underline">
          dxyufei@lnut.edu.cn
        </a>
      </footer>
    </main>
  );
}
