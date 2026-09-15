const { getDefaultConfig } = require('expo/metro-config');
const config = getDefaultConfig(__dirname);
// Bundle the language model and (local-only, git-ignored) code table as raw assets.
config.resolver.assetExts.push('bslm', 'cin');
module.exports = config;
