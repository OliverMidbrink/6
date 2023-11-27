import Image from 'next/image'
import Link from 'next/link';


export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-16 pt-2 lg:p-24">
      <div className="relative text-6xl lg:text-8xlflex place-items-center before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px] z-[-1]">
          <div className='container pt-8'>
            <div><h2 className={`mb-3 text-6xl lg:text-8xl p-6 lg:p-6 pt-8 font-semibold`}>Molytica{' '}</h2></div>
            <div className='box'><p className={`m-0 max-w-[30ch] text-2xl opacity-50`}>AI enabled Drug Discovery Tools</p></div>
          </div>
      </div>
      <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-2 lg:text-left">
        <Link
          href="/multisels"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Molytica M{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            Complex Diseases Require Complex Solutions.
          </p>
        </Link>

        <Link
          href=""
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
        >
          <h2 className={`mb-3 text-2xl font-semibold`}>
            Contact{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            oliver.midbrink@gmail.com
          </p>
          <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
            0763271528
          </p>
        </Link>
      </div>
    </main>
  )
}
