<script setup>
import { ref, onMounted } from 'vue'

import UploadImage from '@/components/UploadImage.vue'
import FooterBar from '@/components/FooterBar.vue'
import IconDownload from '@/components/icons/IconDownload.vue'
import IconTrash from '@/components/icons/IconTrash.vue'
import IconEye from '@/components/icons/IconEye.vue'
import IconEyeClosed from '@/components/icons/IconEyeClosed.vue'
import IconInfo from '@/components/icons/IconInfo.vue'
import HeroSection from '@/components/HeroSection.vue'

import { loadImage, getImageData, normalizeAndTranspose, transformAndTranspose } from '@/utils.js'

import * as ort from 'onnxruntime-web'

// Set the path to the wasm files that are located in the public folder.
// This is required because vite does not handle the wasm files correctly.
// Interested in any better solution to this problem.
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': './ort-wasm.wasm',
  'ort-wasm-simd.wasm': './ort-wasm-simd.wasm',
  'ort-wasm-threaded.wasm': './ort-wasm-threaded.wasm'
}

const uploadedImageSrc = ref(null)
const predictedImageSrc = ref(null)
const activeImageSrc = ref(null)
const isPredicting = ref(false)

async function handleImageUpload(file) {
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader()
    reader.onload = async (e) => {
      uploadedImageSrc.value = e.target.result
      activeImageSrc.value = e.target.result
    }
    reader.readAsDataURL(file)
  }
}

let session = null

async function predict() {
  if (!session || !uploadedImageSrc.value) {
    console.error('Session not initialized or no image uploaded')
    return
  }
  isPredicting.value = true
  const image = await loadImage(uploadedImageSrc.value)

  const imageData = getImageData(image)
  const float32Data = normalizeAndTranspose(imageData, image.width, image.height)

  const tensor = new ort.Tensor('float32', float32Data, [1, 3, image.height, image.width])

  const output = await session.run({ input: tensor })
  const outputImageData = transformAndTranspose(output.output, image.width, image.height)

  const outputImage = new ImageData(outputImageData, image.width, image.height)
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  canvas.width = image.width
  canvas.height = image.height
  ctx.putImageData(outputImage, 0, 0)

  predictedImageSrc.value = canvas.toDataURL()
  activeImageSrc.value = canvas.toDataURL()
  isPredicting.value = false
}

function reset() {
  uploadedImageSrc.value = null
  predictedImageSrc.value = null
  activeImageSrc.value = null
}

function switchImgSrc() {
  if (activeImageSrc.value == uploadedImageSrc.value) {
    activeImageSrc.value = predictedImageSrc.value
  } else {
    activeImageSrc.value = uploadedImageSrc.value
  }
}

function download() {
  const link = document.createElement('a')
  link.href = activeImageSrc.value
  link.download = 'image.png'
  link.click()
}

function imageClicked(src) {
  let base_url = import.meta.env.BASE_URL
  uploadedImageSrc.value = base_url + src
  activeImageSrc.value = base_url + src
}

onMounted(async () => {
  const executionProviders = ['wasm']
  session = await ort.InferenceSession.create('./model.onnx', { executionProviders })
})
</script>

<template>
  <main class="h-screen w-full flex flex-col justify-between items-center overflow-scroll">
    <div class="flex flex-col justify-start items-center">
      <HeroSection>
        <template v-slot:title> Transform Night into Light </template>
        <template v-slot:description>
          Transform your dark photos into bright images directly in your browser. No server uploads,
          ensuring complete privacy.
        </template>
      </HeroSection>
      <UploadImage v-if="!uploadedImageSrc" @upload="handleImageUpload" />
      <div v-else class="flex flex-row justify-center mb-2 items-center">
        <button
          v-if="predictedImageSrc"
          class="text-gray-500 hover:text-gray-900"
          @click="switchImgSrc()"
        >
          <IconEye v-if="activeImageSrc === predictedImageSrc" />
          <IconEyeClosed v-else />
        </button>
        <button
          class="text-gray-500 hover:text-gray-900 mx-3 font-bold bg-gray-200 rounded px-2"
          @click="predict"
        >
          Transform
        </button>
        <button
          v-if="predictedImageSrc"
          class="mr-3 text-gray-500 hover:text-gray-900"
          @click="download()"
        >
          <IconDownload />
        </button>
        <button class="text-gray-500 hover:text-gray-900" @click="reset()">
          <IconTrash />
        </button>
      </div>
      <div
        v-if="!activeImageSrc"
        class="py-8 px-4 mx-auto max-w-screen-xl text-center flex flex-col"
      >
        <p class="mb-0 sm:mb-4 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48">
          Don't have an image click on one of the below to try it out. <br />
          <span class="text-xs"
            >Images taken from
            <a href="https://commons.wikimedia.org/" target="_blank" class="hover:underline"
              >Wikimedia Commons</a
            >.</span
          >
        </p>
        <div class="mt-3 flex flex-col sm:flex-row justify-center items-center">
          <div class="relative">
            <img
              class="max-w-[150px] max-h-[150px] mx-3 mt-3 sm:mt-0 cursor-pointer rounded-lg"
              src="/images/chinatown.jpg"
              @click="imageClicked('/images/chinatown.jpg')"
            />
          </div>
          <img
            class="max-w-[150px] max-h-[150px] mx-3 mt-3 sm:mt-0 cursor-pointer rounded-lg"
            src="/images/livingRoom.jpg"
            @click="imageClicked('/images/livingRoom.jpg')"
          />
          <img
            class="max-w-[150px] max-h-[150px] mx-3 mt-3 sm:mt-0 cursor-pointer rounded-lg"
            src="/images/athens.jpg"
            @click="imageClicked('/images/athens.jpg')"
          />
        </div>
      </div>
      <div class="relative">
        <img
          class="max-w-[330px] lg:max-w-[400px] max-h-[400px] lg:max-h-[400px]"
          :class="isPredicting ? 'opacity-50' : ''"
          v-if="activeImageSrc"
          :src="activeImageSrc"
        />
        <div v-if="isPredicting"  role="status" class="z-20 absolute -translate-x-1/2 -translate-y-1/2 top-2/4 left-1/2">
          <svg
            aria-hidden="true"
            class="w-8 h-8 text-gray-200 animate-spin fill-blue-600"
            viewBox="0 0 100 101"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
              fill="currentColor"
            />
            <path
              d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
              fill="currentFill"
            />
          </svg>
          <span class="sr-only">Loading...</span>
        </div>
      </div>
    </div>
    <FooterBar class="w-full -z-10"></FooterBar>
  </main>
</template>
