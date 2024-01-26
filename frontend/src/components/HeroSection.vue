<template>
  <div class="pt-8 py-4 px-4 mx-auto max-w-screen-xl text-center lg:py-16 lg:px-12">
    <h1
      class="mb-4 mt-4 text-4xl font-extrabold tracking-tight leading-none text-gray-900 md:text-5xl lg:text-6xl"
    >
      <slot name="title"></slot>
    </h1>
    <p class="mb-0 sm:mb-4 text-lg font-normal text-gray-500 lg:text-xl sm:px-16 xl:px-48">
      <slot name="description"></slot>
    </p>
  </div>
</template>
