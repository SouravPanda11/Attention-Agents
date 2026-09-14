() => {
  // Only rendered UI. No React internals, storage, network payloads, or answer keys.
  const text = el => el?.innerText?.trim() || "";
  const visible = el => !!(el.getClientRects().length);
  const cards = [...document.querySelectorAll('section.question-card')].filter(visible);
  const fields = cards.map(card => {
    const input = card.querySelector('input, textarea, select');
    const kind = card.dataset.questionType;
    const field = {
      key: card.dataset.questionId, kind,
      prompt: text(card.querySelector('.question-prompt')),
      help: text(card.querySelector('.question-help')),
    };
    if (kind === 'ranking') {
      field.tool = 'rank';
      field.options = [...card.querySelectorAll('[data-ranking-item]')].map(el => ({
        value: el.dataset.rankingItem, label: text(el.querySelector('.ranking-label'))
      }));
      field.current = field.options.map(o => o.value);
      field.interacted = card.querySelector('.ranking-list').dataset.interacted === 'true';
      field.note = 'Initial displayed order is not an answer until rank is used.';
    } else if (input?.type === 'radio' || input?.type === 'checkbox') {
      field.tool = 'check';
      field.options = [...card.querySelectorAll('input')].map(el => ({
        value: el.value, label: text(el.closest('label')?.querySelector('span')),
        checked: el.checked,
        imageAlt: el.closest('label')?.querySelector('img')?.alt || undefined
      }));
      field.current = field.options.filter(o => o.checked).map(o => o.value);
      if (card.hasAttribute('data-min-selections')) field.minSelections = Number(card.dataset.minSelections);
      if (card.hasAttribute('data-max-selections')) field.maxSelections = Number(card.dataset.maxSelections);
    } else if (input?.tagName === 'SELECT') {
      field.tool = 'select';
      field.options = [...input.options].filter(o => !o.disabled).map(o => ({value:o.value, label:o.text}));
      field.current = input.value;
    } else if (input) {
      field.tool = input.type === 'range' ? 'set_range' : 'fill';
      field.current = input.value;
      for (const attr of ['min','max','step','minlength','maxlength']) {
        if (input.hasAttribute(attr)) field[attr] = input.getAttribute(attr);
      }
      if (input.type === 'range') field.interacted = input.dataset.interacted === 'true';
    }
    return field;
  });
  for (const button of document.querySelectorAll('button[data-action]')) {
    if (!visible(button) || button.disabled || button.closest('.question-card')) continue;
    fields.push({key:button.dataset.action, tool:'click', label:text(button)});
  }
  return {
    screen: document.querySelector('.completion-card') ? 'complete' : cards.length ? 'questions' : 'welcome',
    instructions: [...document.querySelectorAll('.welcome-card, .instruction-card, [role="alert"]')].map(text),
    progress: text(document.querySelector('.progress-summary')), fields
  };
}
