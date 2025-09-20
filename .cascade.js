const fs = require('fs');

module.exports = async function cascade(context) {
  const { taskType, prompt, isMultiFile } = context;

  const selectedModel = (taskType === 'plan' || taskType === 'editPlan' || isMultiFile || prompt.length > 3000)
    ? 'claude-sonnet-4'
    : 'swe-agent-1';

  // Log to console and to file
  const logMsg = `Cascade.js triggered. Task type: ${taskType}, Model: ${selectedModel}\n`;
  console.log("🚨", logMsg);
  fs.appendFileSync('/tmp/cascade-debug.log', logMsg);

  return {
    model: selectedModel,
    maxTokens: 4096,
    autoContinue: true,
    maxInvocations: 5
  };
};