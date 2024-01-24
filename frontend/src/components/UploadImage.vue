<script setup>
const emit = defineEmits(['upload'])

function handleFileChange(event) {
  const file = event.target.files[0]
  if (file) {
    // You can process the file here if needed
    emit('upload', file) // Emit the 'upload' event with the file as payload
  }
}
</script>

<template>
  <div class="flex items-center justify-center w-2/3 sm:w-1/3 h-32">
    <label
      for="dropzone-file"
      class="flex flex-col items-center justify-center w-full h-full border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50"
    >
      <div class="flex flex-col items-center justify-center pt-5 pb-6">
        <svg
          class="w-8 h-8 mb-4 text-gray-500"
          aria-hidden="true"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 20 16"
        >
          <path
            stroke="currentColor"
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
          />
        </svg>
        <p class="mb-2 mx-2 text-sm text-gray-500 text-center">
          <span class="font-semibold">Click to upload</span> or drag and drop a .png, .jpg, .jpeg,
          etc.
        </p>
      </div>
      <input id="dropzone-file" type="file" class="hidden" @change="handleFileChange" />
    </label>
  </div>
</template>