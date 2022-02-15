const { getDefaultConfig } = require('expo/metro-config'); // eslint-disable-line @typescript-eslint/no-var-requires

const config = getDefaultConfig(__dirname);

const defaultAssetExts = config.resolver.assetExts;

config.resolver.assetExts = [...defaultAssetExts, 'bin']; // <-- cjs added here

module.exports = config;
