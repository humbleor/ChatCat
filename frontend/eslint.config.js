// ESLint flat config (ESLint 9+)
// typescript-eslint + eslint-plugin-vue + eslint-config-prettier
import tseslint from 'typescript-eslint';
import pluginVue from 'eslint-plugin-vue';
import prettierConfig from 'eslint-config-prettier/flat';

export default tseslint.config(
  {
    ignores: ['dist/**', 'node_modules/**', 'package-lock.json'],
  },
  ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  {
    // .vue 文件的 <script> 块用 TypeScript parser 解析
    files: ['**/*.vue'],
    languageOptions: {
      parserOptions: {
        parser: tseslint.parser,
        extraFileExtensions: ['.vue'],
        sourceType: 'module',
      },
    },
  },
  {
    // 贴合现有代码风格:any 降级为警告,下划线开头的变量/参数不算未使用
    rules: {
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
      // 允许 References、Sidebar 等单单词组件名
      'vue/multi-word-component-names': 'off',
    },
  },
  // 关掉与 Prettier 冲突的规则(必须放最后)
  prettierConfig
);
