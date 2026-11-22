// Sokoban solver for make.fun's "Build the best Sokoban solver" bounty.
// Strategy: every level is a 1-wide hallway plus 1-4 rooms, each room opening
// onto the hallway through a single door cell, so boxes never leave their room.
// Each room is solved push-optimally on its own with A* over pushes (state =
// sorted box cells + the player's push-free reachable region, canonicalised
// to its lowest cell), an admissible optimal-assignment heuristic (boxes to
// goals by static BFS distances), and deadlock pruning (dead squares, 2x2
// blocks, wall freezes). Rooms are then stitched with plain BFS walks along
// the hallway. All budgets are by work (expansion counts), never the clock,
// so runs are byte-identical. No clock, no randomness, no environment reads.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];
const OPP = [1, 0, 3, 2];
const INF = 1_000_000_000;

// Deterministic work budgets (never the wall clock).
const PER_ROOM_CAP = 600_000;
const GLOBAL_CAP = 3_000_000;
const BEAM_WIDTH = 1500;
const BEAM_DEPTH = 160;
let totalExpansions = 0;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

interface Room {
  w: number; // level width (for 2x2 / freeze tests in global coordinates)
  hall: Uint8Array; // hallway floor cells: the player can stand here, boxes never go here
  cells: number[]; // floor cells for the player: room cells, then the entry cell (if distinct)
  boxable: Uint8Array; // per local cell: 1 when a box may stand here (room cells only)
  local: Int32Array; // global -> local (-1 outside the room)
  wall: Uint8Array; // level wall map (for 2x2 / freeze tests)
  door: number; // global door cell
  doorLocal: number;
  entry: number; // global hallway cell beside the door: where the player starts
  entryLocal: number;
  boxes: number[]; // local, sorted
  goals: number[]; // local
  nbr: Int32Array; // R*4, local neighbour or -1
  isGoal: Uint8Array;
  dist: Int32Array[]; // per goal: BFS distance to every room cell
  live: Uint8Array; // dead-square analysis: 1 = a box here can still reach a goal
  n: number;
  perms: number[][];
}

interface RoomSolution {
  moves: string;
  pushes: number;
  endPlayer: number; // local cell
  endBoxes: number[]; // local, sorted
  exitOK: boolean; // player can reach the door cell in the final state
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#' ? 1 : 0;
      if ('.*+'.includes(ch)) goal[i] = 1;
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// ---------- generic helpers ----------

// BFS walk on the whole level, avoiding boxes. Returns the move string or null.
function walkFull(lv: Level, from: number, to: number, boxes: number[]): string | null {
  if (from === to) return '';
  const boxSet = new Set(boxes);
  const prev = new Int32Array(lv.w * lv.h).fill(-1);
  const prevDir = new Int8Array(lv.w * lv.h).fill(-1);
  const q: number[] = [from];
  prev[from] = from;
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    for (let d = 0; d < 4; d++) {
      const x = c % lv.w;
      if ((d === 2 && x === 0) || (d === 3 && x === lv.w - 1)) continue;
      const nb = c + DY[d]! * lv.w + DX[d]!;
      if (nb < 0 || nb >= lv.w * lv.h || lv.wall[nb]! || prev[nb] !== -1 || boxSet.has(nb)) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        let s = '';
        let cur = to;
        while (cur !== from) {
          s = CH[prevDir[cur]!]! + s;
          cur = prev[cur]!;
        }
        return s;
      }
      q.push(nb);
    }
  }
  return null;
}

// Replay moves on a level. Returns the end state, or null on an illegal step.
function applyMoves(
  lv: Level,
  moves: string,
  player: number,
  boxes: number[],
): { player: number; boxes: number[]; pushes: number } | null {
  const bs = boxes.slice();
  let p = player;
  let pushes = 0;
  for (const raw of moves) {
    const ch = raw.toLowerCase();
    const d = CH.indexOf(ch);
    if (d < 0) continue; // whitespace / newline: ignored
    const x = p % lv.w;
    if ((d === 2 && x === 0) || (d === 3 && x === lv.w - 1)) return null;
    const t = p + DY[d]! * lv.w + DX[d]!;
    if (t < 0 || t >= lv.w * lv.h || lv.wall[t]) return null;
    const at = bs.indexOf(t);
    if (at >= 0) {
      const beyond = t + DY[d]! * lv.w + DX[d]!;
      const bx = t % lv.w;
      if ((d === 2 && bx === 0) || (d === 3 && bx === lv.w - 1)) return null;
      if (beyond < 0 || beyond >= lv.w * lv.h || lv.wall[beyond] || bs.includes(beyond)) return null;
      bs[at] = beyond;
      bs.sort((a, b) => a - b);
      pushes++;
    }
    p = t;
  }
  return { player: p, boxes: bs, pushes };
}

function validate(lv: Level, moves: string): boolean {
  const r = applyMoves(lv, moves, lv.player, lv.boxes);
  return r !== null && r.boxes.every((b) => lv.goal[b] === 1);
}

// ---------- room decomposition ----------

function findHallway(lv: Level): { row: number } | { col: number } | null {
  const py = Math.floor(lv.player / lv.w);
  const px = lv.player % lv.w;
  let rowOk = true;
  for (let x = 1; x < lv.w - 1; x++) if (lv.wall[py * lv.w + x]) rowOk = false;
  if (rowOk) return { row: py };
  let colOk = true;
  for (let y = 1; y < lv.h - 1; y++) if (lv.wall[y * lv.w + px]) colOk = false;
  if (colOk) return { col: px };
  return null;
}

function permsOf(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push(cur.slice());
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

function buildRoom(lv: Level, region: number[], door: number, hall: Uint8Array): Room | null {
  const cells = region.slice().sort((a, b) => a - b);
  // The entry cell: the hallway cell beside the door. The player starts there,
  // and a box sitting on the door can only be pushed from it, so it belongs
  // to the room's player floor (boxes still may never stand on it).
  let entry = door;
  const dgx = door % lv.w;
  for (let d = 0; d < 4; d++) {
    if ((d === 2 && dgx === 0) || (d === 3 && dgx === lv.w - 1)) continue;
    const ng = door + DY[d]! * lv.w + DX[d]!;
    if (ng >= 0 && ng < lv.w * lv.h && hall[ng] === 1) {
      entry = ng;
      break;
    }
  }
  const boxable = new Uint8Array(cells.length + 1);
  for (let i = 0; i < cells.length; i++) boxable[i] = 1;
  if (!cells.includes(entry)) cells.push(entry);
  else boxable[cells.length] = 0;
  const R = cells.length;
  const local = new Int32Array(lv.w * lv.h).fill(-1);
  cells.forEach((c, i) => {
    local[c] = i;
  });
  const boxes = lv.boxes.filter((b) => local[b] >= 0 && boxable[local[b]!] === 1).map((b) => local[b]!);
  const goals = cells.filter((c) => lv.goal[c] === 1 && boxable[local[c]!] === 1).map((c) => local[c]!);
  boxes.sort((a, b) => a - b);
  if (boxes.length === 0 || boxes.length !== goals.length) return null;
  const nbr = new Int32Array(R * 4).fill(-1);
  for (let c = 0; c < R; c++) {
    const g = cells[c]!;
    const gx = g % lv.w;
    for (let d = 0; d < 4; d++) {
      if ((d === 2 && gx === 0) || (d === 3 && gx === lv.w - 1)) continue;
      const ng = g + DY[d]! * lv.w + DX[d]!;
      if (ng >= 0 && ng < lv.w * lv.h) nbr[c * 4 + d] = local[ng]!;
    }
  }
  const isGoal = new Uint8Array(R);
  for (const g of goals) isGoal[g] = 1;
  // Static BFS distances from each goal, ignoring boxes. Boxes only ever
  // move on boxable cells, so the BFS stays on those.
  const dist: Int32Array[] = goals.map((g) => {
    const dd = new Int32Array(R).fill(INF);
    dd[g] = 0;
    const q = [g];
    for (let head = 0; head < q.length; head++) {
      const c = q[head]!;
      for (let d = 0; d < 4; d++) {
        const nb = nbr[c * 4 + d]!;
        if (nb !== -1 && boxable[nb] === 1 && dd[nb] === INF) {
          dd[nb] = dd[c]! + 1;
          q.push(nb);
        }
      }
    }
    return dd;
  });
  // Dead squares: cells from which no goal is reachable by reverse pulls.
  // The pusher cell may be the entry cell (a box on the door is pushed from
  // the hallway), so the check uses the full neighbour table.
  const live = new Uint8Array(R);
  for (const g of goals) live[g] = 1;
  let changed = true;
  while (changed) {
    changed = false;
    for (let c = 0; c < R; c++) {
      if (live[c] || boxable[c] !== 1) continue;
      for (let d = 0; d < 4; d++) {
        const y = nbr[c * 4 + d]!;
        if (y === -1 || !live[y]) continue;
        if (nbr[c * 4 + OPP[d]!]! !== -1) {
          live[c] = 1;
          changed = true;
          break;
        }
      }
    }
  }
  return {
    w: lv.w,
    hall,
    cells,
    boxable,
    local,
    wall: lv.wall,
    door,
    doorLocal: local[door]!,
    entry,
    entryLocal: local[entry]!,
    boxes,
    goals,
    nbr,
    isGoal,
    dist,
    live,
    n: boxes.length,
    perms: permsOf(boxes.length),
  };
}

function buildRooms(lv: Level): Room[] {
  const hw = findHallway(lv);
  if (!hw) return [];
  const isHall = (c: number): boolean =>
    'row' in hw ? Math.floor(c / lv.w) === hw.row : c % lv.w === hw.col;
  const hallMask = new Uint8Array(lv.w * lv.h);
  for (let c = 0; c < lv.w * lv.h; c++) if (!lv.wall[c] && isHall(c)) hallMask[c] = 1;
  const seen = new Uint8Array(lv.w * lv.h);
  const rooms: Room[] = [];
  for (let c = 0; c < lv.w * lv.h; c++) {
    if (lv.wall[c] || isHall(c) || seen[c]) continue;
    let adjHall = false;
    for (let d = 0; d < 4; d++) {
      const x = c % lv.w;
      if ((d === 2 && x === 0) || (d === 3 && x === lv.w - 1)) continue;
      const n = c + DY[d]! * lv.w + DX[d]!;
      if (n >= 0 && n < lv.w * lv.h && !lv.wall[n] && isHall(n)) adjHall = true;
    }
    if (!adjHall) continue;
    // Flood-fill the room from its door cell, never crossing the hallway.
    const region: number[] = [];
    const stack = [c];
    seen[c] = 1;
    while (stack.length > 0) {
      const cur = stack.pop()!;
      region.push(cur);
      for (let d = 0; d < 4; d++) {
        const x = cur % lv.w;
        if ((d === 2 && x === 0) || (d === 3 && x === lv.w - 1)) continue;
        const n = cur + DY[d]! * lv.w + DX[d]!;
        if (n < 0 || n >= lv.w * lv.h || lv.wall[n] || isHall(n) || seen[n]) continue;
        seen[n] = 1;
        stack.push(n);
      }
    }
    const room = buildRoom(lv, region, c, hallMask);
    if (room) rooms.push(room);
  }
  return rooms;
}

// Whole level as a single room (fallback when no hallway is found).
function buildWholeRoom(lv: Level): Room {
  const region: number[] = [];
  for (let c = 0; c < lv.w * lv.h; c++) if (!lv.wall[c]) region.push(c);
  const room = buildRoom(lv, region, lv.player, new Uint8Array(lv.w * lv.h));
  if (!room) throw new Error('unbuildable level');
  return room;
}

// ---------- per-room search primitives ----------

// Flood-fill the player's reachable region (boxes are obstacles). Returns
// the visited mask and the lowest reachable cell (the canonical position).
function floodRegion(room: Room, start: number, boxSet: Set<number>): { seen: Uint8Array; min: number } {
  const R = room.cells.length;
  const seen = new Uint8Array(R);
  const stack = [start];
  seen[start] = 1;
  let min = start;
  while (stack.length > 0) {
    const c = stack.pop()!;
    if (c < min) min = c;
    for (let d = 0; d < 4; d++) {
      const nb = room.nbr[c * 4 + d]!;
      if (nb !== -1 && !seen[nb] && !boxSet.has(nb)) {
        seen[nb] = 1;
        stack.push(nb);
      }
    }
  }
  return { seen, min };
}

function encodeBoxes(boxes: number[], R: number): number {
  let k = 0;
  for (let i = boxes.length - 1; i >= 0; i--) k = k * R + boxes[i]!;
  return k;
}

function decodeBoxes(key: number, n: number, R: number): number[] {
  const b: number[] = [];
  for (let i = 0; i < n; i++) {
    b.push(key % R);
    key = Math.floor(key / R);
  }
  return b;
}

function heuristic(room: Room, boxes: number[]): number {
  let best = INF;
  for (const p of room.perms) {
    let s = 0;
    for (let k = 0; k < room.n; k++) {
      const d = room.dist[p[k]!]![boxes[k]!]!;
      if (d >= INF) {
        s = INF;
        break;
      }
      s += d;
    }
    if (s < best) best = s;
  }
  return best;
}

// Deadlock test for a box configuration (sorted local cells).
function isDead(room: Room, boxes: number[]): boolean {
  const boxSet = new Set(boxes);
  for (const b of boxes) if (!room.live[b]) return true;
  return isDead2x2(room, boxSet) || isFrozen(room, boxes, boxSet);
}

// Blocked for the 2x2 test: a wall, a box, or a floor cell outside this room
// that the player cannot stand on. Hallway cells are NOT blocked: the player
// can stand in the hallway, so a 2x2 touching the hallway is never a trap.
function cellBlocked(room: Room, g: number, boxSet: Set<number>): boolean {
  if (g < 0 || g >= room.wall.length) return true;
  if (room.wall[g]) return true;
  const l = room.local[g]!;
  if (l === -1) return room.hall[g] !== 1;
  return boxSet.has(l);
}

function isDead2x2(room: Room, boxSet: Set<number>): boolean {
  const W = room.w;
  const H = room.wall.length / W;
  for (let y = 0; y + 1 < H; y++) {
    for (let x = 0; x + 1 < W; x++) {
      const cells = [y * W + x, y * W + x + 1, (y + 1) * W + x, (y + 1) * W + x + 1];
      if (!cells.every((g) => cellBlocked(room, g, boxSet))) continue;
      for (const g of cells) {
        const l = room.local[g]!;
        if (l !== -1 && boxSet.has(l) && !room.isGoal[l]) return true;
      }
    }
  }
  return false;
}

// A box against a wall, walled in on both sides along that wall, can never move.
function isFrozen(room: Room, boxes: number[], boxSet: Set<number>): boolean {
  const W = room.w;
  for (const b of boxes) {
    if (room.isGoal[b]) continue;
    const g = room.cells[b]!;
    const gx = g % W;
    for (let d = 0; d < 4; d++) {
      const nx = gx + DX[d]!;
      const ny = Math.floor(g / W) + DY[d]!;
      const ng = ny * W + nx;
      // "Wall ahead" must be a real wall: the player can stand in the hallway,
      // so a box with the hallway behind it can still be pushed back in.
      const wallAhead = nx < 0 || nx >= W || ny < 0 || ng >= room.wall.length || room.wall[ng] === 1;
      if (!wallAhead) continue;
      // Perpendicular neighbours: blocked as push targets when they are walls
      // or floor outside the room (boxes can never be pushed there in a win).
      const p1 = d < 2 ? 2 : 0; // d vertical (0,1) -> left/right (2,3); d horizontal (2,3) -> up/down (0,1)
      const p2 = p1 + 1;
      let blocked = true;
      for (const pd of [p1, p2]) {
        const px = gx + DX[pd]!;
        const py = Math.floor(g / W) + DY[pd]!;
        const pg = py * W + px;
        // Blocked as a push target: a wall, outside the level, or a floor cell
        // a box may never stand on (hallway / another room).
        const l = pg >= 0 && pg < room.wall.length ? room.local[pg]! : -1;
        const wallSide =
          px < 0 || px >= W || py < 0 || pg >= room.wall.length || room.wall[pg] === 1 || l === -1 || room.boxable[l] !== 1;
        if (!wallSide) {
          blocked = false;
          break;
        }
      }
      if (blocked) return true;
    }
  }
  return false;
}

// BFS walk inside a room (boxes are obstacles). Null when unreachable.
function walkRoom(room: Room, from: number, to: number, boxes: number[]): string | null {
  if (from === to) return '';
  const boxSet = new Set(boxes);
  const R = room.cells.length;
  const prev = new Int32Array(R).fill(-1);
  const prevDir = new Int8Array(R).fill(-1);
  const q: number[] = [from];
  prev[from] = from;
  for (let head = 0; head < q.length; head++) {
    const c = q[head]!;
    for (let d = 0; d < 4; d++) {
      const nb = room.nbr[c * 4 + d]!;
      if (nb === -1 || prev[nb] !== -1 || boxSet.has(nb)) continue;
      prev[nb] = c;
      prevDir[nb] = d;
      if (nb === to) {
        let s = '';
        let cur = to;
        while (cur !== from) {
          s = CH[prevDir[cur]!]! + s;
          cur = prev[cur]!;
        }
        return s;
      }
      q.push(nb);
    }
  }
  return null;
}

// Rebuild the full move string from a search's parent pointers.
function reconstruct(
  room: Room,
  parent: number[],
  pushBox: number[],
  pushDir: number[],
  goalIdx: number,
  startBoxes: number[],
  entry: number,
): RoomSolution | null {
  const path: { box: number; dir: number }[] = [];
  for (let t = goalIdx; t > 0; t = parent[t]!) path.push({ box: pushBox[t]!, dir: pushDir[t]! });
  path.reverse();
  const boxes = startBoxes.slice();
  let player = entry;
  let moves = '';
  for (const { box, dir } of path) {
    const pusher = room.nbr[box * 4 + OPP[dir]!]!;
    const w = walkRoom(room, player, pusher, boxes);
    if (w === null) return null;
    moves += w + CH[dir]!;
    const target = room.nbr[box * 4 + dir]!;
    boxes[boxes.indexOf(box)] = target;
    boxes.sort((a, b) => a - b);
    player = box;
  }
  const region = floodRegion(room, player, new Set(boxes));
  return {
    moves,
    pushes: path.length,
    endPlayer: player,
    endBoxes: boxes,
    exitOK: region.seen[room.entryLocal] === 1,
  };
}

// A* over pushes. Returns null when the cap is hit or no solution exists.
function solveRoomAStar(room: Room, requireExit: boolean): RoomSolution | null {
  const R = room.cells.length;
  const n = room.n;
  const startBoxes = room.boxes.slice();
  const entry = room.entryLocal;
  const startCanon = floodRegion(room, entry, new Set(startBoxes)).min;
  const h0 = heuristic(room, startBoxes);
  if (h0 >= INF) return null;

  const sKey: number[] = [];
  const sPlayer: number[] = [];
  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const sG: number[] = [];
  const seen = new Map<number, number>();

  const addState = (boxes: number[], player: number, canon: number, parent: number, pb: number, pd: number, g: number): number => {
    const idx = sKey.length;
    sKey.push(encodeBoxes(boxes, R) * R + canon);
    sPlayer.push(player);
    sParent.push(parent);
    sPushBox.push(pb);
    sPushDir.push(pd);
    sG.push(g);
    return idx;
  };

  // Binary heap of [f, g, idx]; ties prefer deeper g, then lower idx.
  const heap: [number, number, number][] = [];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  const root = addState(startBoxes, entry, startCanon, -1, -1, -1, 0);
  seen.set(sKey[root]!, 0);
  heapPush([h0, 0, root]);
  let expanded = 0;

  while (heap.length > 0) {
    const [, g, s] = heapPop();
    if (seen.get(sKey[s]!) !== g) continue;
    const boxes = decodeBoxes(Math.floor(sKey[s]! / R), n, R);
    const boxSet = new Set(boxes);
    if (boxes.every((b) => room.isGoal[b] === 1)) {
      if (!requireExit) return reconstruct(room, sParent, sPushBox, sPushDir, s, startBoxes, entry);
      const region = floodRegion(room, sPlayer[s]!, boxSet);
      if (region.seen[room.entryLocal] === 1) return reconstruct(room, sParent, sPushBox, sPushDir, s, startBoxes, entry);
      // Otherwise keep searching: this state is solved but the player is trapped.
    }
    if (expanded >= PER_ROOM_CAP || totalExpansions >= GLOBAL_CAP) return null;
    expanded++;
    totalExpansions++;
    const region = floodRegion(room, sPlayer[s]!, boxSet);
    for (const b of boxes) {
      for (let d = 0; d < 4; d++) {
        const target = room.nbr[b * 4 + d]!;
        if (target === -1 || room.boxable[target] !== 1 || boxSet.has(target)) continue;
        const pusher = room.nbr[b * 4 + OPP[d]!]!;
        if (pusher === -1 || !region.seen[pusher]) continue;
        const nb = boxes.slice();
        nb[nb.indexOf(b)] = target;
        nb.sort((a, b2) => a - b2);
        if (isDead(room, nb)) continue;
        const h = heuristic(room, nb);
        if (h >= INF) continue;
        const canon = floodRegion(room, b, new Set(nb)).min;
        const key = encodeBoxes(nb, R) * R + canon;
        const ng = g + 1;
        const prevG = seen.get(key);
        if (prevG !== undefined && prevG <= ng) continue;
        seen.set(key, ng);
        const idx = addState(nb, b, canon, s, b, d, ng);
        heapPush([ng + h, ng, idx]);
      }
    }
  }
  return null;
}

// Beam search fallback (valid, maybe non-optimal). Bounded by width x depth.
function solveRoomBeam(room: Room, requireExit: boolean): RoomSolution | null {
  const R = room.cells.length;
  const n = room.n;
  const startBoxes = room.boxes.slice();
  const entry = room.entryLocal;

  const sParent: number[] = [];
  const sPushBox: number[] = [];
  const sPushDir: number[] = [];
  const sBoxes: number[][] = [];
  const sPlayer: number[] = [];
  const addState = (boxes: number[], player: number, parent: number, pb: number, pd: number): number => {
    const idx = sParent.length;
    sParent.push(parent);
    sPushBox.push(pb);
    sPushDir.push(pd);
    sBoxes.push(boxes);
    sPlayer.push(player);
    return idx;
  };

  const startCanon = floodRegion(room, entry, new Set(startBoxes)).min;
  void startCanon;
  let frontier = [addState(startBoxes, entry, -1, -1, -1)];
  for (let depth = 0; depth < BEAM_DEPTH; depth++) {
    // Goal check at this depth.
    let bestGoal = -1;
    let bestH = INF;
    for (const s of frontier) {
      const boxes = sBoxes[s]!;
      if (!boxes.every((b) => room.isGoal[b] === 1)) continue;
      if (requireExit) {
        const region = floodRegion(room, sPlayer[s]!, new Set(boxes));
        if (region.seen[room.entryLocal] !== 1) continue;
      }
      const h = 0;
      if (h < bestH) {
        bestH = h;
        bestGoal = s;
      }
    }
    if (bestGoal >= 0) return reconstruct(room, sParent, sPushBox, sPushDir, bestGoal, startBoxes, entry);
    const children: { key: number; h: number; idx: number }[] = [];
    const seenKeys = new Set<number>();
    for (const s of frontier) {
      const boxes = sBoxes[s]!;
      const boxSet = new Set(boxes);
      const region = floodRegion(room, sPlayer[s]!, boxSet);
      totalExpansions++;
      for (const b of boxes) {
        for (let d = 0; d < 4; d++) {
          const target = room.nbr[b * 4 + d]!;
          if (target === -1 || room.boxable[target] !== 1 || boxSet.has(target)) continue;
          const pusher = room.nbr[b * 4 + OPP[d]!]!;
          if (pusher === -1 || !region.seen[pusher]) continue;
          const nb = boxes.slice();
          nb[nb.indexOf(b)] = target;
          nb.sort((a, b2) => a - b2);
          if (isDead(room, nb)) continue;
          const h = heuristic(room, nb);
          if (h >= INF) continue;
          const canon = floodRegion(room, b, new Set(nb)).min;
          const key = encodeBoxes(nb, R) * R + canon;
          if (seenKeys.has(key)) continue;
          seenKeys.add(key);
          children.push({ key, h, idx: addState(nb, b, s, b, d) });
        }
      }
    }
    if (children.length === 0) return null;
    children.sort((a, b) => a.h - b.h || a.key - b.key);
    frontier = children.slice(0, BEAM_WIDTH).map((c) => c.idx);
  }
  return null;
}

function solveRoomBest(room: Room, requireExit: boolean): RoomSolution | null {
  return solveRoomAStar(room, requireExit) ?? solveRoomBeam(room, requireExit);
}

// ---------- whole-level solve ----------

function solveLevel(text: string): string {
  const lv = parseLevel(text);
  const rooms = buildRooms(lv);
  if (rooms.length === 0) {
    const room = buildWholeRoom(lv);
    const sol = solveRoomBest(room, false);
    if (!sol) return '';
    return validate(lv, sol.moves) ? sol.moves : '';
  }
  // Visit rooms left to right by door column (walking distance is free).
  const order = rooms
    .map((_, i) => i)
    .sort((a, b) => {
      const da = rooms[a]!.door;
      const db = rooms[b]!.door;
      return da % lv.w - (db % lv.w) || Math.floor(da / lv.w) - Math.floor(db / lv.w);
    });
  const sols: (RoomSolution | null)[] = rooms.map((r) => solveRoomBest(r, false));
  // Every room but the last must let the player back out through its door.
  const triedExit = new Set<number>();
  for (let iter = 0; iter <= rooms.length + 1; iter++) {
    let bad = -1;
    for (let i = 0; i < order.length - 1; i++) {
      const s = sols[order[i]!]!;
      if (!s || !s.exitOK) {
        bad = i;
        break;
      }
    }
    if (bad === -1) break;
    const ri = order[bad]!;
    if (triedExit.has(ri)) {
      order.splice(bad, 1);
      order.push(ri);
      continue;
    }
    triedExit.add(ri);
    const s2 = solveRoomBest(rooms[ri]!, true);
    if (s2) sols[ri] = s2;
    else {
      order.splice(bad, 1);
      order.push(ri);
    }
  }
  if (sols.some((s) => s === null)) return '';
  // Stitch: walk to each door, run its room solution, repeat.
  let moves = '';
  let player = lv.player;
  let boxes = lv.boxes.slice();
  for (const ri of order) {
    const room = rooms[ri]!;
    const sol = sols[ri]!;
    const w = walkFull(lv, player, room.entry, boxes);
    if (w === null) return '';
    moves += w;
    const after = applyMoves(lv, w + sol.moves, player, boxes);
    if (after === null) return '';
    moves += sol.moves;
    player = after.player;
    boxes = after.boxes;
  }
  return validate(lv, moves) ? moves : '';
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
