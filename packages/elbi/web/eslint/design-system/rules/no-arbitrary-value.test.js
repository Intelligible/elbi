// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./no-arbitrary-value.js"

ruleTester.run("no-arbitrary-value", rule, {
  valid: [
    `<p className="text-2xs ring-[3px] w-[340px] border-2" />`,
    `const x = "text-[11px]"`, // not a class context
    `const message = "text-[11px] is small"`, // lowercase, not class-named
    `<p className={cn("text-xs", on && "rounded-md")} />`,
  ],
  invalid: [
    { code: `<p className="text-[11px]" />`, errors: [{ messageId: "fontSize" }] },
    { code: `<p className="md:text-[0.95rem]" />`, errors: [{ messageId: "fontSize" }] },
    { code: `<p className="rounded-[10px]" />`, errors: [{ messageId: "radius" }] },
    { code: `<p className="rounded-tl-[4px]" />`, errors: [{ messageId: "radius" }] },
    { code: `<p className="bg-[#fff]" />`, errors: [{ messageId: "color" }] },
    { code: `<p className="border-[color:var(--x)]" />`, errors: [{ messageId: "color" }] },
    { code: `const fieldClass = "h-8 text-[13px]"`, errors: [{ messageId: "fontSize" }] },
    { code: "<p className={`px-2 text-[10px] ${a}`} />", errors: [{ messageId: "fontSize" }] },
    { code: `<p className={cn({ "text-[11px]": on })} />`, errors: [{ messageId: "fontSize" }] },
    { code: `const TAB_TRIGGER = "text-[11px]"`, errors: [{ messageId: "fontSize" }] },
    { code: `const ICON_14 = "rounded-[3px]"`, errors: [{ messageId: "radius" }] },
    { code: `const triggerClassName = "text-[11px]"`, errors: [{ messageId: "fontSize" }] },
    { code: `<X listClassName="text-[11px]" />`, errors: [{ messageId: "fontSize" }] },
  ],
})
