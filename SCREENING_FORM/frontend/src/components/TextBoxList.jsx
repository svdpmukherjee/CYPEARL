import { Plus, X } from "lucide-react";

export default function TextBoxList({
  items,
  onChange,
  placeholder,
  minBoxes = 1,
  labelPrefix = "",
}) {
  const updateItem = (index, value) => {
    const updated = [...items];
    updated[index] = value;
    onChange(updated);
  };
  const addBox = () => onChange([...items, ""]);
  const removeBox = (index) => {
    if (items.length <= minBoxes) return;
    onChange(items.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {items.map((val, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-xs text-slate-500 shrink-0 font-medium w-16 text-right">
            {labelPrefix ? `${labelPrefix} ${i + 1}` : `${i + 1}.`}
          </span>
          <input
            type="text"
            value={val}
            onChange={(e) => updateItem(i, e.target.value)}
            placeholder={placeholder}
            className="flex-1 border border-slate-300 rounded-md p-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 placeholder:text-slate-400"
          />
          {items.length > minBoxes && (
            <button
              type="button"
              onClick={() => removeBox(i)}
              className="p-1 text-slate-400 hover:text-red-400 transition-colors"
              title="Remove"
            >
              <X size={16} />
            </button>
          )}
        </div>
      ))}
      <button
        type="button"
        onClick={addBox}
        className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 transition-colors mt-1"
      >
        <Plus size={14} /> Add more
      </button>
    </div>
  );
}
