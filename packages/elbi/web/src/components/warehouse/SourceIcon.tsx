// Per-source logos for the warehouse connectors: a real brand logo as an <img> on a
// white chip, so it reads in either theme. Provenance and trademark ownership for every
// image is recorded in public/services/manifest.json. SQLite, which has no image there,
// uses its simple-icons mark; CSV, Custom and Google Cloud Storage use a themed lucide
// glyph, the first two because they are not vendor brands and the third because no mark
// for it is published in the collection the rest came from.

import { Cloud, FileSpreadsheet, type LucideIcon, Puzzle } from "lucide-react"
import { siSqlite } from "simple-icons"

// Connectors with a real logo image in /services/, by file name. A map rather than a set
// because a few of the marks are only published as SVG, and the extension is the whole
// difference: the <img> below renders either.
const LOGO_FILES: Record<string, string> = {
  postgres: "postgres.png",
  mysql: "mysql.png",
  mssql: "mssql.png",
  snowflake: "snowflake.png",
  bigquery: "bigquery.png",
  supabase: "supabase.png",
  redshift: "redshift.png",
  clickhouse: "clickhouse.png",
  cockroachdb: "cockroachdb.png",
  neon: "neon.png",
  oracle: "oracle.png",
  dynamodb: "dynamodb.png",
  elasticsearch: "elasticsearch.png",
  mongodb: "mongodb.svg",
  planetscale: "planetscale.svg",
  s3: "s3.png",
  azure_blob: "azure_blob.png",
  sftp: "sftp.png",
  google_sheets: "google_sheets.svg",
  stripe: "stripe.png",
  hubspot: "hubspot.png",
  salesforce: "salesforce.png",
  shopify: "shopify.png",
  zendesk: "zendesk.png",
  chargebee: "chargebee.png",
  mailchimp: "mailchimp.png",
  klaviyo: "klaviyo.png",
  sendgrid: "sendgrid.png",
  braze: "braze.png",
  pipedrive: "pipedrive.png",
  front: "front.png",
  vercel: "vercel.png",
  airtable: "airtable.png",
  mixpanel: "mixpanel.png",
  github: "github.png",
  jira: "jira.png",
  notion: "notion.png",
  slack: "slack.png",
  sentry: "sentry.png",
  typeform: "typeform.png",
  intercom: "intercom.png",
}

const FALLBACK: Record<string, { icon: LucideIcon; color: string }> = {
  csv: { icon: FileSpreadsheet, color: "#21A366" },
  custom: { icon: Puzzle, color: "#7C3AED" },
  // Google publishes no Cloud Storage mark in the collection the others came from, so
  // this one is a glyph in Google's own storage blue rather than a lookalike.
  gcs: { icon: Cloud, color: "#4285F4" },
}

export function SourceIcon({ type, size = 44 }: { type: string; size?: number }) {
  return (
    <span
      className="inline-flex shrink-0 items-center justify-center"
      style={{ width: size, height: size }}
    >
      <Inner type={type} size={size} />
    </span>
  )
}

function Inner({ type, size }: { type: string; size: number }) {
  const file = LOGO_FILES[type]
  if (file) {
    return (
      <img
        src={`/services/${file}`}
        alt={type}
        className="object-contain"
        style={{ maxWidth: size, maxHeight: size }}
      />
    )
  }
  // No brand image: draw the glyph a touch smaller than the full box.
  const glyph = Math.round(size * 0.86)
  if (type === "sqlite") {
    // currentColor (the surrounding text colour) so it stays legible in both themes
    // without a white backing.
    return (
      <svg
        role="img"
        aria-label="SQLite"
        viewBox="0 0 24 24"
        width={glyph}
        height={glyph}
        fill="currentColor"
      >
        <path d={siSqlite.path} />
      </svg>
    )
  }
  const { icon: Icon, color } = FALLBACK[type] ?? FALLBACK.custom
  return <Icon style={{ width: glyph, height: glyph, color }} />
}
