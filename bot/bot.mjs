#!/usr/bin/env node
// Repo bot: a zero-dependency local agent that uses the local model API
// (OpenAI-compatible) and can read/write files inside this repository.
//
// Usage:
//   node bot/bot.mjs "解释 app.js 的结构"
//   node bot/bot.mjs "在 README.md 末尾加一行测试说明" --write
//   MODEL_API_URL=http://127.0.0.1:18801/v1 MODEL_NAME=opencode/deepseek-v4.1-flash-free node bot/bot.mjs "..."
import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
let cfg = {}
try {
  cfg = JSON.parse(fs.readFileSync(path.join(HERE, 'config.json'), 'utf8'))
} catch {}
let BRAIN = ''
try {
  BRAIN = fs.readFileSync(path.join(HERE, 'brain.md'), 'utf8').trim()
} catch {}

const ROOT = process.cwd()
const API = (process.env.MODEL_API_URL || cfg.api || 'http://127.0.0.1:18801/v1').replace(/\/$/, '')
const MODEL = process.env.MODEL_NAME || cfg.model || 'opencode/deepseek-v4.1-flash-free'
const MAX_TURNS = Number(process.env.BOT_MAX_TURNS || cfg.max_turns || 6)
const TEMPERATURE = Number(cfg.temperature ?? 0.2)
const ALLOW_WRITE = process.argv.includes('--write')
const args = process.argv.slice(2).filter((a) => !a.startsWith('--'))
const task = args.join(' ')
if (!task) {
  console.error('用法: node bot/bot.mjs "任务描述" [--write]')
  process.exit(1)
}

function within(p) {
  const abs = path.resolve(ROOT, p || '.')
  if (abs !== ROOT && !abs.startsWith(ROOT + path.sep)) throw new Error('路径越界: ' + p)
  return abs
}

const tools = [
  {
    type: 'function',
    function: {
      name: 'list_dir',
      description: '列出仓库内某个目录的文件',
      parameters: { type: 'object', properties: { path: { type: 'string' } }, required: [] }
    }
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: '读取仓库内文本文件（可指定行数）',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' }, limit: { type: 'integer' } },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: '写入仓库内文件（需要 --write 参数才会启用）',
      parameters: {
        type: 'object',
        properties: { path: { type: 'string' }, content: { type: 'string' } },
        required: ['path', 'content']
      }
    }
  }
].filter((t) => ALLOW_WRITE || t.function.name !== 'write_file')

function runTool(name, a) {
  if (name === 'list_dir') {
    const dir = within(a.path || '.')
    return fs.readdirSync(dir, { withFileTypes: true })
      .map((d) => (d.isDirectory() ? d.name + '/' : d.name))
      .join('\n')
  }
  if (name === 'read_file') {
    const p = within(a.path)
    const text = fs.readFileSync(p, 'utf8')
    const lines = text.split(/\r?\n/)
    const n = Number(a.limit) > 0 ? Number(a.limit) : lines.length
    return lines.slice(0, n).join('\n')
  }
  if (name === 'write_file') {
    const p = within(a.path)
    fs.mkdirSync(path.dirname(p), { recursive: true })
    fs.writeFileSync(p, a.content ?? '', 'utf8')
    return 'written: ' + path.relative(ROOT, p)
  }
  return 'ERROR: unknown tool ' + name
}

async function chat(messages) {
  const res = await fetch(API + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer local' },
    body: JSON.stringify({
      model: MODEL,
      messages,
      tools,
      tool_choice: 'auto',
      max_tokens: 2048,
      temperature: TEMPERATURE
    }),
    signal: AbortSignal.timeout(180000)
  })
  const j = await res.json()
  if (!res.ok) throw new Error(`model api ${res.status}: ${JSON.stringify(j).slice(0, 200)}`)
  return j.choices?.[0]?.message ?? {}
}

const messages = [
  {
    role: 'system',
    content:
      BRAIN ||
      '你是仓库里的本地机器人。使用工具查看/修改仓库文件，任务完成后用中文给出结论。只操作当前仓库内的路径。'
  },
  { role: 'user', content: task }
]

for (let turn = 1; turn <= MAX_TURNS; turn++) {
  const msg = await chat(messages)
  const calls = msg.tool_calls || []
  if (!calls.length) {
    console.log(msg.content || '(空回复)')
    process.exit(0)
  }
  messages.push({ role: 'assistant', content: msg.content || '', tool_calls: calls })
  for (const c of calls) {
    let out
    try {
      const a = JSON.parse(c.function.arguments || '{}')
      out = runTool(c.function.name, a)
    } catch (e) {
      out = 'ERROR: ' + e.message
    }
    console.error(`[tool] ${c.function.name} -> ${String(out).slice(0, 120).replace(/\n/g, ' ')}`)
    messages.push({ role: 'tool', tool_call_id: c.id, content: String(out).slice(0, 20000) })
  }
}
console.error('达到最大轮数，停止。')
