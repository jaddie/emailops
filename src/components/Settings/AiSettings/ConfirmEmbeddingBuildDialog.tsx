import { useTranslation } from 'react-i18next';
import type { EmbeddingBuildPreview } from '@/lib/api';
import { useOverlay } from '@/stores/overlayStore';

export function ConfirmEmbeddingBuildDialog({
  preview,
  onDecide,
}: {
  preview: EmbeddingBuildPreview;
  onDecide: (yes: boolean) => void;
}) {
  useOverlay();
  const { t } = useTranslation(['common', 'settings']);
  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="embedding-confirm-title"
    >
      <div className="bg-white border border-gray-300 rounded-lg p-6 shadow-xl max-w-md w-full mx-4">
        <h3 id="embedding-confirm-title" className="text-base font-semibold mb-2">
          {t('settings:embeddingLifecycle.title')}
        </h3>
        <p className="text-sm mb-3">
          {t('settings:embeddingLifecycle.confirm', {
            model: preview.profile.model,
            dimensions: preview.profile.dimensions,
          })}
        </p>
        <p className="text-sm mb-3 break-all">
          {preview.profile.backend} {preview.profile.endpoint} ·{' '}
          {preview.profile.model_revision ?? t('settings:embeddingLifecycle.noRevision')}
        </p>
        <p className="text-sm mb-3">{t('settings:embeddingLifecycle.warning')}</p>
        {preview.opaqueIdentity && <p className="text-sm mb-3">{t('settings:embeddingLifecycle.identity')}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" onClick={() => onDecide(false)}>
            {t('common:actions.cancel')}
          </button>
          <button
            type="button"
            className="px-3 py-1.5 bg-primary-600 text-white rounded"
            onClick={() => onDecide(true)}
          >
            {t('settings:embeddingLifecycle.activate')}
          </button>
        </div>
      </div>
    </div>
  );
}
