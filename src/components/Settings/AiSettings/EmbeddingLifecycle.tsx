import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Select } from '@/components/shared/Select';
import * as api from '@/lib/api';
import { errorText } from '@/lib/errors';

interface Props {
  policy: api.EmbeddingDimensionPolicy;
  onPolicy: (policy: api.EmbeddingDimensionPolicy) => void;
  onBuild: () => void;
  onResume?: (model: string, policy: api.EmbeddingDimensionPolicy) => void;
  onRollback?: () => void;
  busy: boolean;
}

export function EmbeddingLifecycle({ policy, onPolicy, onBuild, onResume, onRollback, busy }: Props) {
  const { t } = useTranslation('settings');
  const [lifecycle, setLifecycle] = useState<api.EmbeddingLifecycle | null>(null);
  const [error, setError] = useState('');
  const [acting, setActing] = useState(false);
  const mounted = useRef(false);
  const requestSequence = useRef(0);
  const refresh = useCallback(async () => {
    const sequence = ++requestSequence.current;
    try {
      const value = await api.getEmbeddingLifecycle();
      if (mounted.current && sequence === requestSequence.current) {
        setLifecycle(value);
        setError('');
      }
    } catch (err) {
      if (mounted.current && sequence === requestSequence.current) setError(errorText(err));
    }
  }, []);
  useEffect(() => {
    mounted.current = true;
    void refresh();
    const timer = setInterval(() => void refresh(), 3000);
    return () => {
      mounted.current = false;
      ++requestSequence.current;
      clearInterval(timer);
    };
  }, [refresh]);
  const action = async (name: 'stop' | 'resume' | 'discard' | 'rollback', generation: number | null) => {
    if (!lifecycle) return;
    ++requestSequence.current;
    setActing(true);
    setError('');
    try {
      await api.controlEmbeddingBuild(name, generation, lifecycle.revision);
      await refresh();
    } catch (err) {
      ++requestSequence.current;
      if (mounted.current) setError(errorText(err));
    } finally {
      if (mounted.current) setActing(false);
    }
  };
  return (
    <section className="p-4 rounded-lg border border-gray-200 bg-gray-50 space-y-3">
      <h3 className="text-sm font-semibold">{t('embeddingLifecycle.title')}</h3>
      <p className="text-xs text-gray-600">{t('embeddingLifecycle.warning')}</p>
      <p className="text-xs text-gray-600">{t('embeddingLifecycle.identity')}</p>
      {(['active', 'pending', 'retained'] as const).map((kind) => {
        const entry = lifecycle?.[kind];
        if (!entry) return null;
        return (
          <div key={kind} className="text-sm border-t pt-2">
            <strong>{t(`embeddingLifecycle.${kind}`)}:</strong>{' '}
            {t(`embeddingLifecycle.state.${entry.status.state}`, { defaultValue: entry.status.state })} ·{' '}
            {entry.profile?.model ?? t('embeddingLifecycle.legacy')} · {t('embeddingLifecycle.detected')}:{' '}
            {entry.status.dimensions}
            {entry.profile && (
              <div className="text-xs break-all">
                {t('embeddingLifecycle.requested')}:{' '}
                {entry.profile.dimension_policy === 'Native'
                  ? t('embeddingLifecycle.native')
                  : entry.profile.dimension_policy.Explicit}{' '}
                · {entry.profile.backend} {entry.profile.endpoint} ·{' '}
                {entry.profile.model_revision ?? t('embeddingLifecycle.noRevision')} · {entry.status.fingerprint}
              </div>
            )}
            {entry.corpora.map((c) => (
              <div key={c.corpus} className="text-xs">
                {t(`embeddingLifecycle.${c.corpus}`)}: {c.completed} / {c.completed + c.pending} ·{' '}
                {t('embeddingLifecycle.failed')}: {c.failed}{' '}
                {c.requestedEnabled ? '' : t('embeddingLifecycle.disabled')}
              </div>
            ))}
            {entry.status.error && <p className="text-red-700">{entry.status.error}</p>}
          </div>
        );
      })}
      <div className="flex gap-2 items-center text-sm">
        <span>{t('embeddingLifecycle.dimensions')}</span>
        <Select
          testId="embedding-policy"
          ariaLabel={t('embeddingLifecycle.dimensions')}
          value={policy === 'Native' ? 'native' : 'explicit'}
          options={[
            { value: 'native', label: t('embeddingLifecycle.native') },
            { value: 'explicit', label: t('embeddingLifecycle.explicit') },
          ]}
          onChange={(value) => onPolicy(value === 'native' ? 'Native' : { Explicit: 768 })}
        />
        {policy !== 'Native' && (
          <input
            aria-label={t('embeddingLifecycle.dimensions')}
            type="number"
            min="1"
            max="8192"
            value={policy.Explicit}
            onChange={(e) => onPolicy({ Explicit: Number(e.target.value) })}
            className="w-24 border rounded"
          />
        )}
      </div>
      {error && (
        <p className="text-sm text-red-700" role="alert">
          {error}
        </p>
      )}
      <div className="flex gap-2 flex-wrap text-sm">
        <button type="button" disabled={busy || acting || !!lifecycle?.pending} onClick={onBuild}>
          {t('embeddingLifecycle.preview')}
        </button>
        {lifecycle?.pending &&
          (['building', 'validating', 'preparing'].includes(lifecycle.pending.status.state) ? (
            <button
              type="button"
              disabled={acting}
              onClick={() => void action('stop', lifecycle.pending?.status.id ?? null)}
            >
              {t('embeddingLifecycle.stop')}
            </button>
          ) : (
            <>
              <button
                type="button"
                disabled={acting}
                onClick={() => {
                  if (onResume && lifecycle.pending?.profile)
                    onResume(lifecycle.pending.profile.model, lifecycle.pending.profile.dimension_policy);
                  else void action('resume', lifecycle.pending?.status.id ?? null);
                }}
              >
                {t('embeddingLifecycle.resume')}
              </button>
              <button
                type="button"
                disabled={acting}
                onClick={() => void action('discard', lifecycle.pending?.status.id ?? null)}
              >
                {t('embeddingLifecycle.discard')}
              </button>
            </>
          ))}
        {lifecycle?.retained && !lifecycle.pending && (
          <button
            type="button"
            disabled={acting}
            onClick={() => (onRollback ? onRollback() : void action('rollback', null))}
          >
            {t('embeddingLifecycle.rollback')}
          </button>
        )}
      </div>
    </section>
  );
}
