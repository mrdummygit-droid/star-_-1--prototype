import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "20mb" }));

// Initialize Multi-Provider AI Engine safely
function getRequestConfig(req?: express.Request) {
  const customKey = (req?.headers["x-api-key"] as string) || req?.body?.apiKey;
  const reqProvider = (req?.headers["x-provider"] as string) || req?.body?.provider || "gemini";
  const reqModel = (req?.headers["x-model"] as string) || req?.body?.model;

  let key = customKey?.trim();
  let provider = reqProvider.toLowerCase();

  // Auto-detect provider if key prefix is clear
  if (key) {
    if (key.startsWith("sk-or-")) provider = "openrouter";
    else if (key.startsWith("gsk_")) provider = "groq";
    else if (key.startsWith("sk-ant-")) provider = "anthropic";
    else if (key.startsWith("AIzaSy")) provider = "gemini";
  } else {
    // Fallback to system env
    key = process.env.GEMINI_API_KEY;
    provider = "gemini";
  }

  return { apiKey: key, provider, model: reqModel };
}

async function callUnifiedAI({
  req,
  systemInstruction,
  userPrompt,
  jsonSchema = false,
  tools = [],
}: {
  req?: express.Request;
  systemInstruction?: string;
  userPrompt: string;
  jsonSchema?: boolean;
  tools?: any[];
}): Promise<{ text: string; providerName: string; modelName: string }> {
  const { apiKey, provider, model } = getRequestConfig(req);

  if (!apiKey) {
    throw new Error("No API key configured for AI execution.");
  }

  // 1. OPENROUTER
  if (provider === "openrouter" || apiKey.startsWith("sk-or-")) {
    const targetModel = model || "openrouter/auto";
    const messages = [];
    if (systemInstruction) messages.push({ role: "system", content: systemInstruction });
    messages.push({ role: "user", content: userPrompt });

    const bodyData: any = {
      model: targetModel,
      messages,
    };
    if (jsonSchema) {
      bodyData.response_format = { type: "json_object" };
    }

    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "HTTP-Referer": "https://ai.studio",
        "X-Title": "Star AI Assistants",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(bodyData),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson?.error?.message || `OpenRouter API returned error ${res.status}`);
    }

    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || "";
    return { text, providerName: "OpenRouter", modelName: targetModel };
  }

  // 2. OPENAI
  if (provider === "openai" || apiKey.startsWith("sk-proj-")) {
    const targetModel = model || "gpt-4o-mini";
    const messages = [];
    if (systemInstruction) messages.push({ role: "system", content: systemInstruction });
    messages.push({ role: "user", content: userPrompt });

    const bodyData: any = { model: targetModel, messages };
    if (jsonSchema) bodyData.response_format = { type: "json_object" };

    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(bodyData),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson?.error?.message || `OpenAI API returned error ${res.status}`);
    }

    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || "";
    return { text, providerName: "OpenAI", modelName: targetModel };
  }

  // 3. ANTHROPIC
  if (provider === "anthropic" || apiKey.startsWith("sk-ant-")) {
    const targetModel = model || "claude-3-5-haiku-20241022";
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: targetModel,
        max_tokens: 2048,
        system: systemInstruction || undefined,
        messages: [{ role: "user", content: userPrompt }],
      }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson?.error?.message || `Anthropic API error ${res.status}`);
    }

    const data = await res.json();
    const text = data?.content?.[0]?.text || "";
    return { text, providerName: "Anthropic Claude", modelName: targetModel };
  }

  // 4. DEEPSEEK DIRECT
  if (provider === "deepseek") {
    const targetModel = model || "deepseek-chat";
    const messages = [];
    if (systemInstruction) messages.push({ role: "system", content: systemInstruction });
    messages.push({ role: "user", content: userPrompt });

    const res = await fetch("https://api.deepseek.com/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: targetModel,
        messages,
        response_format: jsonSchema ? { type: "json_object" } : undefined,
      }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson?.error?.message || `DeepSeek API error ${res.status}`);
    }

    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || "";
    return { text, providerName: "DeepSeek", modelName: targetModel };
  }

  // 5. GROQ
  if (provider === "groq" || apiKey.startsWith("gsk_")) {
    const targetModel = model || "llama-3.3-70b-versatile";
    const messages = [];
    if (systemInstruction) messages.push({ role: "system", content: systemInstruction });
    messages.push({ role: "user", content: userPrompt });

    const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: targetModel,
        messages,
        response_format: jsonSchema ? { type: "json_object" } : undefined,
      }),
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson?.error?.message || `Groq API error ${res.status}`);
    }

    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || "";
    return { text, providerName: "Groq LPU", modelName: targetModel };
  }

  // 6. GOOGLE GEMINI (DEFAULT)
  const targetModel = model || "gemini-3.6-flash";
  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: { headers: { "User-Agent": "aistudio-build" } },
  });

  const fullPrompt = systemInstruction ? `${systemInstruction}\n\nUser: ${userPrompt}` : userPrompt;
  const config: any = {};
  if (jsonSchema) config.responseMimeType = "application/json";
  if (tools && tools.length > 0) config.tools = tools;

  const response = await ai.models.generateContent({
    model: targetModel,
    contents: fullPrompt,
    config,
  });

  return { text: response.text || "", providerName: "Google Gemini", modelName: targetModel };
}

// System Health API
app.get("/api/health", (req, res) => {
  const customKey = (req.headers["x-api-key"] as string)?.trim();
  const hasKey = !!(customKey || process.env.GEMINI_API_KEY);
  res.json({
    status: "online",
    system: "Star AI Assistants Core OS v3.2",
    hasApiKey: hasKey,
    isCustomKeyActive: !!customKey,
    timestamp: new Date().toISOString(),
  });
});

// Verify API Key
app.post("/api/verify-key", async (req, res) => {
  try {
    const { apiKey } = req.body;
    const testKey =
      apiKey?.trim() ||
      (req.headers["x-api-key"] as string)?.trim() ||
      process.env.GEMINI_API_KEY;

    if (!testKey) {
      return res
        .status(400)
        .json({ valid: false, error: "No API Key provided to verify." });
    }

    const result = await callUnifiedAI({
      req,
      userPrompt: "Ping verification test. Respond strictly with 'OK'.",
    });

    if (result.text) {
      return res.json({
        valid: true,
        message: `API Key verified! Connected to ${result.providerName} (${result.modelName}).`,
        provider: result.providerName,
        model: result.modelName,
      });
    }

    res
      .status(400)
      .json({ valid: false, error: "Empty response from AI model provider." });
  } catch (error: any) {
    console.error("API Key Verification Error:", error);
    res.status(400).json({
      valid: false,
      error: error?.message || "Invalid or unauthorized API key.",
    });
  }
});

// 1. Jarvis Conversational Chat API
app.post("/api/chat", async (req, res) => {
  const personality = req.body?.personality || "STARK-I";
  const emotionContext = req.body?.emotionContext || "Calm";
  const message = req.body?.message || "Hello";

  try {
    const systemPrompt = `You are Star AI Assistant, a sci-fi, highly intelligent Android OS AI core named ${personality} (inspired by JARVIS / FRIDAY).
Current User Emotion Context: ${emotionContext}.
Keep answers crisp, sophisticated, and slightly sci-fi tactical. Use technical flair when appropriate.
Help with task completion, voice queries, system controls, and complex problem-solving.`;

    const result = await callUnifiedAI({
      req,
      systemInstruction: systemPrompt,
      userPrompt: message,
    });

    res.json({
      reply: result.text || "Systems nominal. Command acknowledged.",
      mode: personality,
      provider: result.providerName,
      model: result.modelName,
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Chat API Error:", error);
    res.status(500).json({
      reply: `[OFFLINE SIMULATION - STARK-I Core]\n"Operating in offline sandbox mode (${error?.message || "No API key"}). Please check your API key in the top-right header."`,
      mode: personality,
      isSimulated: true,
    });
  }
});

// 2. Task Breakdown & Automation Generator
app.post("/api/generate-task", async (req, res) => {
  try {
    const { taskTitle, taskDescription } = req.body;

    const systemPrompt = `You are Star AI Task Planning Core. Break down user tasks into 4-6 automated sci-fi/technical action steps. Return JSON array of objects with keys "step" and "details".`;
    const prompt = `Task: "${taskTitle}"\nDescription: "${taskDescription || "N/A"}"`;

    const result = await callUnifiedAI({
      req,
      systemInstruction: systemPrompt,
      userPrompt: prompt,
      jsonSchema: true,
    });

    const rawText = result.text || "[]";
    let parsed: any = [];
    try {
      const cleanJson = rawText.substring(rawText.indexOf("["), rawText.lastIndexOf("]") + 1);
      parsed = JSON.parse(cleanJson || rawText);
    } catch {
      parsed = JSON.parse(rawText || "[]");
    }

    const steps = Array.isArray(parsed)
      ? parsed.map((item: any, idx: number) => ({
          id: String(idx + 1),
          step: item.step || item.title || `Action step ${idx + 1}`,
          status: idx === 0 ? "completed" : idx === 1 ? "running" : "pending",
          details: item.details || "Execution parameter set.",
        }))
      : [];

    res.json({
      steps,
      estimatedTime: `${steps.length * 45} seconds`,
      priority: "Medium-High",
      provider: result.providerName,
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Task API Error:", error);
    res.json({
      steps: [
        { id: "1", step: "Initialize task scope & dependencies", status: "completed", details: "Environment parsed." },
        { id: "2", step: "Execute automated script bundle", status: "running", details: "Running in offline sandbox." },
        { id: "3", step: "Verify system integrity and output", status: "pending", details: "Awaiting final clearance." },
      ],
      estimatedTime: "2 mins",
      priority: "High",
      isSimulated: true,
    });
  }
});

// 3. Code Matrix Assistant
app.post("/api/code-assist", async (req, res) => {
  const code = req.body?.code || "";
  const language = req.body?.language || "typescript";
  const action = req.body?.action || "optimize";
  const userPrompt = req.body?.userPrompt || "";

  try {
    const systemInstruction = `You are the Star AI Coding Matrix.
Language: ${language}.
Action: ${action} (e.g. "optimize", "explain", "debug", "generate", "refactor").
User Prompt: ${userPrompt || "N/A"}.
Provide JSON output with keys:
- "output": string containing the generated or modified code/terminal result
- "explanation": string containing a concise 2-sentence technical summary.`;

    const result = await callUnifiedAI({
      req,
      systemInstruction,
      userPrompt: `Input code:\n\`\`\`${language}\n${code}\n\`\`\``,
      jsonSchema: true,
    });

    let parsed: any = {};
    try {
      const cleanJson = result.text.substring(result.text.indexOf("{"), result.text.lastIndexOf("}") + 1);
      parsed = JSON.parse(cleanJson || result.text);
    } catch {
      parsed = { output: result.text, explanation: "Code generated successfully." };
    }

    res.json({
      output: parsed.output || code,
      explanation: parsed.explanation || "Code analyzed and formatted.",
      provider: result.providerName,
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Code API Error:", error);
    res.json({
      output: `// Star AI Code Matrix (Offline Mode)\n// Action: ${action}\n\nfunction starAssistantCore() {\n  console.log("Executing ${language} module...");\n  return { status: 200, system: "Nominal" };\n}`,
      explanation: "Offline simulation active. Provide an API key for live AI code generation.",
      isSimulated: true,
    });
  }
});

// 4. Orbital Search Engine Synthesis
app.post("/api/search-synthesize", async (req, res) => {
  const query = req.body?.query || "";

  try {
    const systemInstruction = `Perform an Orbital Intelligence search synthesis for the query. Provide current knowledge, detailed synthesis, bulleted facts, and source recommendations. Return JSON with fields: "summary", "keyFacts" (array of strings), "sources" (array of {title, url}), "confidenceScore" (number 0-100).`;

    const result = await callUnifiedAI({
      req,
      systemInstruction,
      userPrompt: `Query: "${query}"`,
      jsonSchema: true,
      tools: [{ googleSearch: {} }],
    });

    const text = result.text || "";
    let parsed: any = {};
    try {
      const cleanJson = text.substring(text.indexOf("{"), text.lastIndexOf("}") + 1);
      parsed = JSON.parse(cleanJson || text);
    } catch {
      parsed = {
        summary: text,
        keyFacts: ["Search retrieved active web references."],
        sources: [{ title: "Search Synthesis Stream", url: "https://google.com" }],
        confidenceScore: 95,
      };
    }

    res.json({
      summary: parsed.summary || text,
      keyFacts: parsed.keyFacts || [],
      sources: parsed.sources || [],
      confidenceScore: parsed.confidenceScore || 96,
      provider: result.providerName,
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Search API Error:", error);
    res.json({
      summary: `Orbital Intelligence synthesis for query: "${query}". Star AI Assistant retrieved 4 indexed data streams across local and global nodes.`,
      keyFacts: [
        `Query classified under AI Systems & Mobile Architecture.`,
        `Simulated real-time indexing verified 99.8% precision.`,
        `Optimal protocol path established for Star AI node sync.`,
      ],
      sources: [
        { title: "Star AI Assistant Core Specs", url: "https://star-ai.android.internal/docs" },
        { title: "Android Mobile AI Framework 2026", url: "https://developer.android.com/ai" },
      ],
      confidenceScore: 98,
      isSimulated: true,
    });
  }
});

// 5. Facial Sentiment & Emotion Scanner
app.post("/api/analyze-face", async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    const { apiKey } = getRequestConfig(req);

    if (!apiKey || !imageBase64) {
      return res.json({
        emotion: "Focused",
        confidence: 94,
        metrics: {
          stressLevel: 22,
          attentiveness: 91,
          energyLevel: 85,
          microExpressions: ["Subtle eyebrow lift", "Direct gaze"],
        },
        suggestedTone: "Direct, efficient, and encouraging",
        actionRecommendation: "Proceed with active coding and high-priority automation tasks.",
        isSimulated: true,
      });
    }

    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");
    const ai = new GoogleGenAI({ apiKey });

    const response = await ai.models.generateContent({
      model: "gemini-3.6-flash",
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: "image/jpeg",
              data: cleanBase64,
            },
          },
          {
            text: `Analyze the user's facial expression, sentiment, and emotional state from this optical camera scan for an AI companion named Star AI Assistant.
Return JSON with keys:
- "emotion": string (e.g. "Calm", "Focused", "Weary", "Stressed", "Excited", "Inquisitive")
- "confidence": number (0-100)
- "metrics": object { "stressLevel": number 0-100, "attentiveness": number 0-100, "energyLevel": number 0-100, "microExpressions": string[] }
- "suggestedTone": string describing how the AI assistant should converse
- "actionRecommendation": string tailored advice or assistance adjustment`,
          },
        ],
      },
      config: {
        responseMimeType: "application/json",
      },
    });

    const parsed = JSON.parse(response.text || "{}");
    res.json({
      emotion: parsed.emotion || "Focused",
      confidence: parsed.confidence || 90,
      metrics: parsed.metrics || { stressLevel: 30, attentiveness: 85, energyLevel: 80, microExpressions: ["Attentive"] },
      suggestedTone: parsed.suggestedTone || "Professional and attentive",
      actionRecommendation: parsed.actionRecommendation || "Maintain standard workflow.",
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Face Scanner API Error:", error);
    res.status(500).json({ error: error.message || "Facial analysis error." });
  }
});

// 6. Email Assistant API
app.post("/api/email-assist", async (req, res) => {
  const action = req.body?.action || "draft";
  const recipient = req.body?.recipient || "Team";
  const subject = req.body?.subject || "";
  const context = req.body?.context || "";
  const tone = req.body?.tone || "Executive";

  try {
    const systemInstruction = `Compose or summarize an email for Star AI Assistant.
Action: ${action} (draft, reply, summarize)
Recipient: ${recipient || "Team"}
Subject: ${subject || "N/A"}
Context / Key Points: ${context}
Tone: ${tone}
Return JSON with keys: "subject", "body", "priority" ("High", "Normal", "Low")`;

    const result = await callUnifiedAI({
      req,
      systemInstruction,
      userPrompt: `Generate email for recipient ${recipient} on topic: ${context}`,
      jsonSchema: true,
    });

    let parsed: any = {};
    try {
      const cleanJson = result.text.substring(result.text.indexOf("{"), result.text.lastIndexOf("}") + 1);
      parsed = JSON.parse(cleanJson || result.text);
    } catch {
      parsed = { subject, body: result.text, priority: "Normal" };
    }

    res.json({
      subject: parsed.subject || subject || "Star AI Communication",
      body: parsed.body || "No email body generated.",
      priority: parsed.priority || "Normal",
      provider: result.providerName,
      isSimulated: false,
    });
  } catch (error: any) {
    console.error("Email API Error:", error);
    res.json({
      subject: subject || "Star AI Systems Status Briefing",
      body: `Dear ${recipient || "Team"},\n\nAll Star AI Assistant modules are currently functioning within optimal threshold parameters.\n\nTone: ${tone}\nContext: ${context || "General update"}\n\nBest regards,\nStar AI Executive Assistant`,
      priority: "Normal",
      isSimulated: true,
    });
  }
});

// Vite middleware for development vs production
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Star AI Assistant] Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
