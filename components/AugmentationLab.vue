<script setup lang="ts">
import { computed, ref } from 'vue'

type Task = 'classification' | 'detection' | 'segmentation'
type Mode = 'original' | 'flip' | 'crop' | 'color'
type Box = { name: string, color: string, coordinates: number[] }

const task = ref<Task>('detection')
const mode = ref<Mode>('original')
const boxes: Box[] = [
  { name: 'cat', color: '#ed7d31', coordinates: [18, 22, 72, 82] },
  { name: 'dog', color: '#8e6bbf', coordinates: [94, 28, 145, 83] },
]
const tasks: { id: Task, label: string }[] = [
  { id: 'classification', label: 'Classification' },
  { id: 'detection', label: 'Detection' },
  { id: 'segmentation', label: 'Segmentation' },
]
const modes: { id: Mode, label: string }[] = [
  { id: 'original', label: 'Original' },
  { id: 'flip', label: 'Horizontal flip' },
  { id: 'crop', label: 'Shared crop' },
  { id: 'color', label: 'Color jitter' },
]

const viewBox = computed(() => mode.value === 'crop' ? '0 10 85 80' : '0 0 160 100')
const sceneTransform = computed(() => mode.value === 'flip' ? 'translate(160 0) scale(-1 1)' : '')
const sceneStyle = computed(() => mode.value === 'color'
  ? { filter: 'hue-rotate(65deg) saturate(1.8) brightness(1.12)' }
  : {})

const transformedBoxes = computed(() => boxes.flatMap(box => {
  let [x1, y1, x2, y2] = box.coordinates
  if (mode.value === 'flip') [x1, x2] = [160 - x2, 160 - x1]
  if (mode.value === 'crop') {
    x1 = Math.max(0, Math.min(85, x1))
    x2 = Math.max(0, Math.min(85, x2))
    y1 = Math.max(0, Math.min(80, y1 - 10))
    y2 = Math.max(0, Math.min(80, y2 - 10))
  }
  return x2 > x1 && y2 > y1 ? [{ ...box, coordinates: [x1, y1, x2, y2] }] : []
}))

const targetReadout = computed(() => {
  if (task.value === 'classification') return 'label = “cat”'
  if (task.value === 'segmentation') {
    return mode.value === 'color'
      ? 'mask unchanged: same class IDs at the same pixels'
      : 'image and mask share one geometric transform'
  }
  if (!transformedBoxes.value.length) return 'targets = []'
  return transformedBoxes.value.map(box =>
    `${box.name}: [${box.coordinates.map(Math.round).join(', ')}]`,
  ).join(' · ')
})

function selectTask(value: Task) {
  task.value = value
}
</script>

<template>
  <div class="aug-lab">
    <div class="aug-controls">
      <div class="aug-group" aria-label="Computer vision task">
        <button
          v-for="item in tasks" :key="item.id" :data-task="item.id"
          :class="{ selected: task === item.id }" @click="selectTask(item.id)"
        >{{ item.label }}</button>
      </div>
      <div class="aug-group" aria-label="Augmentation">
        <button
          v-for="item in modes" :key="item.id" :data-mode="item.id"
          :class="{ selected: mode === item.id }" @click="mode = item.id"
        >{{ item.label }}</button>
      </div>
    </div>

    <div class="aug-panels" :class="{ single: task !== 'segmentation' }">
      <div>
        <div class="aug-title">Input image</div>
        <svg :viewBox="viewBox" role="img" aria-label="Synthetic scene after the selected augmentation">
          <g :transform="sceneTransform" :style="sceneStyle">
            <rect width="160" height="100" fill="#a9d2ec" />
            <rect y="60" width="160" height="40" fill="#8dbb72" />
            <g fill="#b67a48">
              <ellipse cx="43" cy="55" rx="24" ry="17" />
              <circle cx="61" cy="39" r="13" />
              <polygon points="51,31 51,17 60,29" />
              <polygon points="62,28 70,17 71,34" />
              <rect x="27" y="64" width="7" height="21" />
              <rect x="50" y="64" width="7" height="21" />
              <path d="M22 49 L8 32" stroke="#b67a48" stroke-width="6" />
              <circle cx="65" cy="38" r="1.8" fill="#202020" />
            </g>
            <g v-if="task !== 'classification' && mode !== 'crop'" fill="#665043">
              <ellipse cx="119" cy="59" rx="23" ry="16" />
              <circle cx="136" cy="45" r="12" />
              <polygon points="128,37 128,25 136,35" />
              <polygon points="138,35 145,25 145,40" />
              <rect x="103" y="67" width="7" height="19" />
              <rect x="126" y="67" width="7" height="19" />
              <path d="M98 54 L88 39" stroke="#665043" stroke-width="6" />
              <circle cx="140" cy="44" r="1.8" fill="#202020" />
            </g>
          </g>
          <g v-if="task === 'detection'">
            <template v-for="box in transformedBoxes" :key="box.name">
              <rect
                :x="box.coordinates[0]"
                :y="box.coordinates[1] + (mode === 'crop' ? 10 : 0)"
                :width="box.coordinates[2] - box.coordinates[0]"
                :height="box.coordinates[3] - box.coordinates[1]"
                fill="none" :stroke="box.color" stroke-width="2.2"
              />
              <text
                :x="box.coordinates[0] + 2"
                :y="box.coordinates[1] + (mode === 'crop' ? 18 : 8)"
                :fill="box.color" font-size="5" font-weight="700"
              >{{ box.name }}</text>
            </template>
          </g>
        </svg>
      </div>

      <div v-if="task === 'segmentation'">
        <div class="aug-title">Target mask</div>
        <svg :viewBox="viewBox" role="img" aria-label="Segmentation target after the same geometric transform">
          <g :transform="sceneTransform">
            <rect width="160" height="100" fill="#9cc9eb" />
            <rect y="60" width="160" height="40" fill="#7cb66b" />
            <g fill="#ed7d31">
              <ellipse cx="43" cy="55" rx="24" ry="17" />
              <circle cx="61" cy="39" r="13" />
              <polygon points="51,31 51,17 60,29" />
              <polygon points="62,28 70,17 71,34" />
              <rect x="27" y="64" width="7" height="21" />
              <rect x="50" y="64" width="7" height="21" />
              <path d="M22 49 L8 32" stroke="#ed7d31" stroke-width="6" />
            </g>
            <g v-if="mode !== 'crop'" fill="#8e6bbf">
              <ellipse cx="119" cy="59" rx="23" ry="16" />
              <circle cx="136" cy="45" r="12" />
              <polygon points="128,37 128,25 136,35" />
              <polygon points="138,35 145,25 145,40" />
              <rect x="103" y="67" width="7" height="19" />
              <rect x="126" y="67" width="7" height="19" />
              <path d="M98 54 L88 39" stroke="#8e6bbf" stroke-width="6" />
            </g>
          </g>
        </svg>
      </div>

      <div v-else-if="task === 'detection'" class="aug-target-box">
        <div class="aug-title">Targets after clipping and empty-box removal</div>
        <div v-for="box in transformedBoxes" :key="box.name" class="aug-box-row">
          <span :style="{ background: box.color }" />{{ box.name }} · xyxy [{{ box.coordinates.map(Math.round).join(', ') }}]
        </div>
        <div v-if="!transformedBoxes.length">No objects remain</div>
      </div>
      <div v-else class="aug-target-box aug-classification">
        <div class="aug-title">Target</div>
        cat
      </div>
    </div>

    <div id="target-readout" class="aug-readout">{{ targetReadout }}</div>
    <div class="aug-hint">Try Detection → Shared crop: the dog leaves both the image and the target list. Color jitter affects only the image.</div>
  </div>
</template>

<style scoped>
.aug-lab { color: #23395d; font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size: 13px; }
.aug-controls { display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; margin-bottom: 8px; }
.aug-group { display: flex; gap: 5px; flex-wrap: wrap; }
.aug-group button { border: 1px solid #c9d1dc; border-radius: 5px; background: white; color: #40516a; padding: 4px 9px; }
.aug-group button.selected { background: #2a78d6; border-color: #2a78d6; color: white; }
.aug-panels { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: center; }
.aug-panels.single { grid-template-columns: 1.55fr 1fr; }
.aug-title { color: #6a7585; font-size: 12px; font-weight: 600; margin-bottom: 4px; }
.aug-panels svg { width: 100%; height: 265px; display: block; overflow: hidden; background: #f4f6f8; border-radius: 6px; }
.aug-target-box { min-height: 100px; padding: 13px; border-radius: 7px; background: #f4f6f8; line-height: 1.6; }
.aug-box-row { margin: 6px 0; font-variant-numeric: tabular-nums; }
.aug-box-row span { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.aug-classification { text-align: center; font-size: 26px; font-weight: 700; }
.aug-classification .aug-title { font-size: 12px; font-weight: 600; }
.aug-readout { margin-top: 7px; padding: 7px 10px; border-left: 4px solid #ed7d31; background: #fff7ef; font-variant-numeric: tabular-nums; }
.aug-hint { margin-top: 5px; color: #777; text-align: center; font-size: 11.5px; }
</style>
